### 一、键缓存（Key Cache）量化方法：通道混合精度量化（2-bit+1-bit+FFT）  
#### 1. **问题建模与变量定义**  
- 设键缓存矩阵 \( K \in \mathbb{R}^{l \times d} \)，其中 \( l \) 为token数，\( d \) 为通道数（隐藏层维度）。  
- 每个通道 \( c \in \{1, 2, ..., d\} \) 对应一列向量 \( K_c \in \mathbb{R}^l \)。  
- **目标**：对每个通道 \( c \) 决定量化精度（2-bit或1-bit），最小化量化误差。  

#### 2. **通道分组：异常通道筛选（2-bit量化）**  
- **筛选标准**：计算通道范围 \( \text{Range}_c = \max(K_c) - \min(K_c) \)，选择前 \( k\% \)（如50%）高Range通道为异常通道（集合 \(\mathcal{D}\)），剩余为正常通道（集合 \(\mathcal{N}\)）。  
- **量化公式（2-bit组量化）**：  
  - 缩放因子：$$ s_c = \frac{\text{Range}_c}{2^2 - 1} = \frac{\text{Range}_c}{3} $$  
  - 偏移量：$$ z_c = \min(K_c) $$  
  - 量化映射：$$ Q(K_c) = \left\lfloor \frac{K_c - z_c}{s_c} \right\rfloor \in \{0, 1, 2, 3\} $$（整数化）  
  - 反量化：$$ K'_c = Q(K_c) \cdot s_c + z_c $$  

#### 3. **正常通道处理（1-bit量化+FFT）**  
- **频域转换（FFT）**：  
  对 \( c \in \mathcal{N} \)，将时域信号 \( K_c \) 转换为频域：  
  $$ K_{c,\text{fft}} = \text{FFT}(K_c) = A + iB \quad (A, B \in \mathbb{R}^l, i为虚数单位) $$  
  其中 \( A \) 为实部，\( B \) 为虚部，频域信号波动更平滑（图3）。  

- **1-bit符号量化**：  
  - 实部量化：$$ Q(A) = \text{sign}(A) = \begin{cases} 1, & A \geq 0 \\ 0, & A < 0 \end{cases} $$  
  - 虚部量化：$$ Q(B) = \text{sign}(B) = \begin{cases} 1, & B \geq 0 \\ 0, & B < 0 \end{cases} $$  
  - 合并表示：用2-bit存储（实部1-bit，虚部1-bit），对应符号组合 \(\{00, 01, 10, 11\}\)。  

- **幅度恢复**：  
  计算频域幅度均值作为缩放因子：  
  $$ s_{\text{fft}} = \text{mean}\left( \left| K_{c,\text{fft}} \right| \right) = \text{mean}\left( \sqrt{A^2 + B^2} \right) $$  
  反量化时通过逆FFT恢复时域信号：  
  $$ K'_c = \text{IFFT}\left( (2Q(A) - 1) + i(2Q(B) - 1) \right) \cdot s_{\text{fft}} $$  
  其中 \( 2Q(\cdot) - 1 \) 将 \(\{0, 1\}\) 映射为 \(\{-1, 1\}\)，模拟频域符号。  


### 二、值缓存（Value Cache）量化方法：1.58-bit三元量化+语义保护（STP）  
#### 1. **基础1.58-bit三元量化（非保护token）**  
- **变量定义**：值缓存矩阵 \( V \in \mathbb{R}^{l \times d} \)，每个元素 \( v_{i,c} \) 表示第 \( i \) 个token的第 \( c \) 个通道值。  
- **阈值计算**：  
  对每个通道 \( c \)，计算绝对值均值：  
  $$ \mu_c = \text{mean}\left( \left| V_c \right| \right), \quad \alpha_c = \gamma \cdot \mu_c \quad (\gamma=0.7 \text{为超参数}) $$  
- **三元映射规则**：  
  $$ Q(v_{i,c}) = \begin{cases} 1, & v_{i,c} > \alpha_c, \\ -1, & v_{i,c} < -\alpha_c, \\ 0, & \text{否则} \end{cases} $$  
- **存储方式**：每个值用2-bit编码（1→11，-1→10，0→00），实际占用 \(\log_2(3) \approx 1.58\) 位。  

#### 2. **语义保护机制（STP：20%关键token保留2-bit）**  
- **关键token筛选**：  
  计算视觉token \( v_i \in \mathbb{R}^d \) 与文本查询token \( t_j \in \mathbb{R}^d \) 的交叉注意力分数（余弦相似度简化）：  
  $$ \mathcal{I}(v_i) = \sum_{j=1}^{m} v_i \cdot t_j^\top \quad (m \text{为文本token数}) $$  
  取前 \( p\% = 20\% \) 的token构成保护集 \(\mathcal{E}\)。  

- **2-bit组量化（保护token）**：  
  对 \( i \in \mathcal{E} \)，每个通道 \( c \) 独立计算：  
  $$ \text{Range}_{i,c} = \max(V_{i,c}) - \min(V_{i,c}), \quad s_{i,c} = \frac{\text{Range}_{i,c}}{3}, \quad z_{i,c} = \min(V_{i,c}) $$  
  量化映射：  
  $$ Q(v_{i,c}) = \left\lfloor \frac{v_{i,c} - z_{i,c}}{s_{i,c}} \right\rfloor \in \{0, 1, 2, 3\} $$  
  反量化：$$ v'_{i,c} = Q(v_{i,c}) \cdot s_{i,c} + z_{i,c} $$  

- **混合精度计算**：  
  保护token占比20%，使用2-bit（4级别）；非保护token占80%，使用1.58-bit（3级别）。  
  平均比特率：$$ 0.2 \times 2 + 0.8 \times 1.58 = 1.664 \text{位} $$（近似1.66位）。  


### 三、量化流程详细步骤（以VideoLLM为例）  
#### **键缓存量化流程**  
1. **输入键矩阵**：\( K \in \mathbb{R}^{l \times d} \)（\( l \) 个token，\( d \) 个通道）。  
2. **通道分组**：  
   - 对每个通道 \( c \)，计算 \( \text{Range}_c = \max(K_c) - \min(K_c) \)。  
   - 按Range降序排序，选择前 \( k\% \) 通道为异常通道 \(\mathcal{D}\)，剩余为正常通道 \(\mathcal{N}\)（如 \( k=50 \)）。  
3. **异常通道处理（2-bit）**：  
   - 对 \( c \in \mathcal{D} \)，计算 \( s_c = \text{Range}_c / 3 \)，\( z_c = \min(K_c) \)。  
   - 量化：\( Q(K_c) = \left\lfloor (K_c - z_c) / s_c \right\rfloor \)，存储为2-bit整数。  
4. **正常通道处理（1-bit+FFT）**：  
   - 对 \( c \in \mathcal{N} \)，执行FFT：\( K_{c,\text{fft}} = \text{FFT}(K_c) \)，分解为实部 \( A \) 和虚部 \( B \)。  
   - 符号量化：\( Q(A) = \text{sign}(A) \)，\( Q(B) = \text{sign}(B) \)（1-bit表示符号）。  
   - 计算 \( s_{\text{fft}} = \text{mean}(|K_{c,\text{fft}}|) \)，存储符号位和缩放因子。  
5. **输出**：量化后的键缓存，异常通道存2-bit整数，正常通道存1-bit符号位+缩放因子。  

#### **值缓存量化流程**  
1. **输入值矩阵**：\( V \in \mathbb{R}^{l \times d} \)。  
2. **关键token筛选（STP）**：  
   - 计算视觉token与文本查询的交叉注意力分数 \( \mathcal{I}(v_i) = v_i \cdot X_t^\top \)（\( X_t \) 为文本token矩阵）。  
   - 取前20%的token索引为 \(\mathcal{E}\)，剩余为 \(\mathcal{\~E}\)。  
3. **保护token处理（2-bit，按通道）**：  
   - 对 \( i \in \mathcal{E} \)，每个通道 \( c \)：  
     \( \text{Range}_{i,c} = \max(V_{i,c}) - \min(V_{i,c}) \)，\( s_{i,c} = \text{Range}_{i,c}/3 \)，\( z_{i,c} = \min(V_{i,c}) \)  
     \( Q(V_{i,c}) = \left\lfloor (V_{i,c} - z_{i,c}) / s_{i,c} \right\rfloor \)（2-bit整数）。  
4. **非保护token处理（1.58-bit，按通道）**：  
   - 对每个通道 \( c \)，计算 \( \mu_c = \text{mean}(|V_c|) \)，\( \alpha_c = 0.7 \cdot \mu_c \)。  
   - 按三元规则量化 \( v_{i,c} \in \mathcal{\~E} \) 为 \(-1, 0, 1\)，存储为2-bit编码。  
5. **合并输出**：按通道维度存储，保护token用2-bit，非保护用1.58-bit，附带通道级缩放因子和偏移量。  


### 四、核心公式推导与变量说明  
#### 1. **键缓存异常通道2-bit量化**  
- **缩放因子推导**：2-bit有 \( 2^2 = 4 \) 个量化级别（0-3），覆盖范围为 \(\text{Range}_c\)，故每个级别代表 \( \text{Range}_c / 3 \)（因0对应最小值，3对应最大值，共3个间隔）。  
- **变量**：  
  - \( K_c \): 通道 \( c \) 的所有token值（向量）。  
  - \( s_c, z_c \): 通道级缩放因子和偏移量，唯一确定该通道的量化映射。  

#### 2. **正常通道FFT处理**  
- **频域优势**：时域信号 \( x(t) \) 的高频成分导致剧烈波动，FFT转换为频域 \( X(f) \) 后，能量集中在低频，高频成分少，符号量化（仅保留符号）对低频信息损失小。  
- **逆变换恢复**：通过逆FFT（IFFT）将符号化的频域信号恢复为时域，幅度由均值 \( s_{\text{fft}} \) 缩放，补偿量化损失。  

#### 3. **值缓存三元量化阈值**  
- **\(\alpha_c = \gamma \cdot \text{mean}(|V_c|)\)**：通过超参数 \(\gamma\) 调整阈值，实验发现 \(\gamma=0.7\) 时性能最佳（图7），平衡零值保留与非零值区分。  
- **计算优势**：三元值的矩阵乘法可转换为加减法，如 \( w \cdot 1 = w \)，\( w \cdot (-1) = -w \)，\( w \cdot 0 = 0 \)，大幅减少计算量（图4）。  


### 五、与LLM量化方法（KIVI）的核心区别  
| **模块**       | **LLM（KIVI）**                          | **VideoLLM（VidKV）**                      |  
|----------------|-----------------------------------------|-------------------------------------------|  
| **键缓存分组** | 所有通道统一2-bit，Per-Channel           | 异常通道2-bit，正常通道1-bit+FFT（通道混合）|  
| **值缓存分组** | Per-Token（每个token独立量化）           | Per-Channel（所有token按通道统一量化）      |  
| **异常处理**   | 无筛选，统一量化                         | 按Range筛选异常通道，针对性高精度处理      |  
| **频域应用**   | 未使用                                  | 正常通道FFT降低时域波动，提升1-bit可行性    |  
| **语义保护**   | 无                                      | 基于注意力分数筛选关键token，保留2-bit精度  |  


### 六、量化误差与性能平衡  
- **键缓存**：异常通道用2-bit避免范围过大导致的截断误差，正常通道FFT后1-bit量化误差降低50%（图3），因频域分布更平滑。  
- **值缓存**：保护20%关键token（约占总比特的8%）使平均比特仅增加0.08位，却能显著提升精度（表2），因关键token的语义信息被完整保留。  

通过这种“精准投放”的混合精度策略，VidKV在极低比特率下（1.5/1.58位）实现了与FP16几乎相同的性能，核心在于对视频数据分布的深度适配——利用通道级冗余压缩正常部分，用更高精度保护异常值和关键语义信息。  