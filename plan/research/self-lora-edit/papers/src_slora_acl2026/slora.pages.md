# Extracted pages: slora.pdf

## Page 1

SLoRA: Balancing Plasticity and Forgetting in Large Language
                          Models for Continual Learning

                       Lina Yang, Yusheng Liao, Yanfeng Wang, Yu Wang*
                                  Shanghai Jiao Tong University
              {alina_yln, liao20160907, wangyanfeng, yuwangsjtu}@sjtu.edu.cn



                           Abstract                               user preferences, LLMs are expected to acquire
                                                                  new capabilities over time (Winata et al., 2023).
        Large language models (LLMs) have achieved
                                                                  This continual learning (CL) setting poses a signifi-
        remarkable success across diverse tasks
        through large-scale pretraining. However, they
                                                                  cant challenge: catastrophic forgetting (McCloskey
        remain prone to catastrophic forgetting in con-           and Cohen, 1989), where learning new tasks in-
        tinual learning. To the best of our knowledge,            terferes with previously acquired knowledge and
        this is the first work to identify noise accumu-          degrades performance on earlier tasks.
        lation in LoRA updates as a key cause of for-                To alleviate the catastrophic forgetting phe-
        getting in continual learning. A preliminary              nomenon, classical CL approaches attempt to add
        two-task experiment demonstrates that remov-              regularization during the training process (Kirk-
        ing less important components of the second
                                                                  patrick et al., 2017; Zenke et al., 2017; Wang
        task’s LoRA parameters improves performance
        on the first task, suggesting that later updates          et al., 2023a; Wu et al., 2024b). For example,
        introduce noisy interference. Building on this            GEM (Lopez-Paz and Ranzato, 2017) constrains
        insight, we propose Subspace-Denoised Low-                updates using gradients stored from previous tasks,
        Rank Adaptation (SLoRA), a simple and effec-              and O-LoRA (Wang et al., 2023a) restricts param-
        tive framework that filters noisy components              eter updates to an orthogonal subspace. However,
        from LoRA updates via subspace similarity                 we found that such regularization prevents forget-
        with the base model. SLoRA is a regularization-
                                                                  ting at the cost of reducing the LLMs’ ability to
        free method without accessing data or gradients
        from previous tasks or modifying the training
                                                                  learn new tasks (as shown in Sec. 6.2). Recent
        process. It offers two variants, SLoRA-Pre                methods like SD-LoRA (Wu et al., 2025), while
        and SLoRA-Post, for online and offline con-               not explicitly imposing regularization, still restrict
        tinual learning, respectively. Extensive exper-           the learning of parameters at the subspace level,
        iments across tasks and models validate the               which is considered a form of implicit constraint.
        effectiveness of SLoRA. It improves final accu-           These findings lead to a question: Can we miti-
        racy by up to 12%, reduces forgetting by 29%,             gate catastrophic forgetting without restricting the
        and filters out over 30% of LoRA parameters
                                                                  LLMs’ capacity for learning?
        identified as noisy. Our code is available at
        https://github.com/alina1031/SLoRA.                          In light of this challenge, we consider a
                                                                  regularization-free method to circumvent the limita-
   1    Introduction                                              tion of the regularization-based methods in restrict-
                                                                  ing the LLMs’ learning capacity. This methodol-
   Large language models (LLMs) have revolution-                  ogy is motivated by prior work showing that update
   ized artificial intelligence, achieving unprecedented          parameters contain a significant amount of redun-
   performance in reasoning, generation, and gener-               dancy and noise (Zhang et al., 2025; Jiang et al.,
   alization across diverse tasks (Brown et al., 2020;            2024, 2025). We hypothesize that the less impor-
   Achiam et al., 2023; Chowdhery et al., 2023). How-             tant components of the new task are noise, which
   ever, these models are typically trained under static          interfere with the knowledge retained from previ-
   task distributions and lack mechanisms for contin-             ous tasks, leading to catastrophic forgetting. To val-
   ual adaptation (Dhingra et al., 2022; Jang et al.,             idate this hypothesis, we conduct a controlled pre-
   2022). In real-world scenarios, such as personal-              experiment where an LLM is sequentially trained
   ized assistants, domain-specific agents, or evolving           on the Amazon Reviews (Task1) and Yahoo An-
       * Corresponding author.                                    swers (Task2) (details are shown in Appendix B.3).
                                                             5437
Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5437–5454
                                July 2-7, 2026 ©2026 Association for Computational Linguistics

## Page 2

70                                                             tion in LoRA as a key factor causing catas-
                     DropFirst              DropAll
           65        DropSecond             Seq-LoRA                      trophic forgetting. To address this, we pro-
           60                                                             pose a method to remove noise from LoRA pa-
           55                                                             rameters, significantly reducing interference

Accuracy
                                                                          caused by the following updated noise.
           50
           45                                                            • Based on this finding, we propose SLoRA,
           40                                                              a novel regularization-free method with two
           35                                                              variants (SLoRA-Pre and SLoRA-Post). Our
           30                                                              approach is efficient and practical as it re-
                10   20   30      40   50     60   70   80   90
                               Drop Rate (%)                               quires no past data or gradients and no modi-
                                                                           fications to the training process.
 Figure 1: Performance of Task1 under LoRA dropping
 with Seq-LoRA as the reference, including DropFirst,                    • We empirically demonstrate that SLoRA out-
 DropSecond, and DropAll. DropFirst degrades perfor-
                                                                           performs all baselines on various continual
 mance, while DropSecond yields clear improvements.
                                                                           learning tasks. Notably, on the TRACE bench-
                                                                           mark, SLoRA improves final accuracy by up
 After training, we proportionally reduce the less im-                     to 12% and reduces forgetting by up to 29%
 portant components in the LoRA parameters using                           by removing over 30% of LoRA parameters.
 three conditions: DropFirst (from Task1), DropSec-
 ond (from Task2), and DropAll (from both), rang-                    2    Related Works
 ing from 10% to 90%. As shown in Figure 1,
 dropping a moderate portion of Task2 parameters                     Continual Learning Continual Learning aims
 raises Task1 accuracy above the Seq-LoRA ref-                       to enable models to learn from a sequence of
 erence, whereas DropFirst consistently degrades                     tasks while retaining previously acquired knowl-
 Task1. This indicates that LoRA updates from later                  edge and minimizing catastrophic forgetting (Mc-
 tasks introduce noisy components that interfere                     Closkey and Cohen, 1989; Ratcliff, 1990; Wu et al.,
 with knowledge retained from earlier tasks, lead-                   2024a). Existing methods are broadly categorized
 ing to our key insight: catastrophic forgetting can                 into three families: (1) Rehearsal-based meth-
 be mitigated by removing noisy components from                      ods (Rolnick et al., 2019; Lopez-Paz and Ranzato,
 LoRA updates to reduce cross-task interference.                     2017; Han et al., 2020) utilize memory buffers or
    Motivated by this insight, we propose Subspace-                  generative models to revisit data from prior tasks,
 Denoised Low-Rank Adaptation (SLoRA), a novel                       which often introduces significant memory or com-
 framework designed to mitigate catastrophic for-                    putational overhead. (2) Regularization-based
 getting by selectively removing noisy components                    methods (Kirkpatrick et al., 2017; Li and Hoiem,
 based on subspace similarity. At its core, SLoRA                    2017; Zenke et al., 2017) prevent forgetting by
 utilizes Singular Value Decomposition on each                       penalizing updates to important parameters. No-
 LoRA update. It then retains components that ex-                    table examples include Elastic Weight Consolida-
 hibit higher similarity to the base model’s represen-               tion (EWC) (Kirkpatrick et al., 2017), which esti-
 tation space, treating low-similarity components as                 mates parameter importance using the Fisher Infor-
 noise to be discarded. We instantiate two distinct                  mation Matrix, and Orthogonal Gradient Descent
 variants: SLoRA-Pre, which performs incremental                     (OGD) (Farajtabar et al., 2020), which constrains
 denoising after each task, making it ideal for on-                  new gradients to be orthogonal to those of prior
 line continual learning settings; and SLoRA-Post,                   tasks. (3) Architecture-based approaches (Rusu
 which performs batch denoising after all tasks,                     et al., 2016; Yoon et al., 2018; Li et al., 2019) iso-
 suitable for offline settings. Crucially, SLoRA is                  late task-specific components to mitigate interfer-
 lightweight, requires no rehearsal data or storage                  ence, such as Progressive Prompts (Razdaibiedina
 of past gradients, and integrates seamlessly without                et al., 2023) which assign soft prompts per task.
 any changes to the standard training process.                       While effective in small-scale settings, these clas-
    In summary, our contributions are as follows:                    sical approaches often struggle with scalability or
                                                                     parameter growth in high-dimensional LLM envi-
           • We are the first to identify noise accumula-            ronments.
                                                                  5438

## Page 3

Parameter-Efficient Fine-Tuning PEFT has                   previously learned tasks {T1 , . . . , Tk−1 }, ensuring
emerged as a practical alternative to full fine-tuning     that the model parameters evolve without suffering
by adapting only a small, carefully selected subset        from catastrophic forgetting.
of parameters (Houlsby et al., 2019; Zaken et al.,
                                                           Low-Rank Adaptation for Continual Learning.
2022; Ding et al., 2023). Beyond adapters (Houlsby
                                                           Given the immense computational cost of full fine-
et al., 2019) and prompt-based tuning (Li and
                                                           tuning, we adopt LoRA (Hu et al., 2021) as the
Liang, 2021; Lester et al., 2021), LoRA (Hu et al.,
                                                           backbone for efficient adaptation. We start with a
2021) has gained significant popularity by inject-
                                                           pretrained LLM parameterized by frozen weights
ing trainable low-rank matrices into frozen weights.
                                                           θ0 . LoRA injects trainable rank decomposition ma-
Recent variants further optimize this process by dy-
                                                           trices into a specific subset of modules, denoted by
namically allocating rank budgets based on weight                                                           ′
                                                           M. For a specific module m ∈ M, let θ0m ∈ Rd ×d
sensitivity (AdaLoRA) (Zhang et al., 2023) or lever-
                                                           represent its frozen base weight. The adaptation
aging principal and minor singular components
                                                           for the k-th task is parameterized by a low-rank
(Pissa, MiLoRA) (Meng et al., 2024; Wang et al.,                                                           d′ ×r
                                                           update ∆θkm = Bkm Am      k , where Bk ∈ R
                                                                                                  m
2024; Liang and Li, 2024). Despite their efficiency,
                                                           and Am k ∈ R
                                                                          r×d are trainable matrices with rank
standard PEFT updates still suffer from interfer-
                                                           r ≪ min(d′ , d).
ence and forgetting when applied to sequential task
                                                              In a sequential fine-tuning setting (Seq-LoRA),
adaptation.
                                                           the model parameters are updated incrementally.
PEFT for Continual Learning Increasingly,                  After training on T tasks, the final adapted weight
PEFT methods are extended to CL to achieve effi-           for module m is obtained by accumulating the up-
cient adaptation while maintaining stability (Song         dates from all tasks:
et al., 2023; Qiao and Mahdavi, 2024). For ex-                                   T
                                                                                 X
ample, O-LoRA (Wang et al., 2023a) enforces or-                    θTm = θ0m +         ∆θkm ,   m ∈ M,        (1)
thogonality between LoRA updates across tasks,                                   k=1
while LB-CL (Qiao and Mahdavi, 2024) combines
low-rank decomposition with sensitivity-based ini-         where ∆θkm denotes the raw LoRA update learned
tialization. ConPET (Song et al., 2023) introduces         for task Tk . Although mathematically concise, this
a memory replay mechanism to manage parameter              direct accumulation often leads to interference, as
reuse. While concurrent work LoRI (Zhang et al.,           ∆θkm may contain components that conflict with the
2025) explores subspace filtering to reduce cross-         knowledge encoded in θ0m or prior updates. Unlike
task interference in multi-task learning, it applies       previous works that add regularization terms to the
filtering during training with access to all tasks.        training objective to constrain ∆θkm , our method,
In contrast, our SLoRA is a regularization-free            SLoRA, focuses on identifying and filtering out
framework that applies subspace-based denoising            noisy components within ∆θkm directly via sub-
post-hoc. Unlike prior methods relying on exter-           space analysis, as detailed in the following section.
nal modules such as prompt routing (Wang et al.,
2022), reasoning traces (Wang et al., 2023b), or
                                                           4     SLoRA: Subspace-Denoised LoRA
teacher-student distillation (Zhong et al., 2024),         4.1    Method Overview
SLoRA filters noisy components directly via sub-           SLoRA is a post-hoc denoising framework de-
space similarity with the base model, offering a           signed to improve the robustness of continual learn-
simple, modular, and scalable solution for long            ing by removing noisy components from task-
task sequences.                                            specific updates. As illustrated in Figure 2, it in-
                                                           tegrates into standard LoRA fine-tuning pipelines
3   Preliminaries
                                                           without modifying the base model or accessing to
Continual Learning Setup. We consider a stan-              data and gradients from previous tasks.
dard continual learning scenario where a pretrained           Consider a pretrained LLM with frozen param-
LLM is adapted to a sequence of tasks T =                  eters θ0 and a set of target modules M. For each
{T1 , T2 , . . . , TT }. Each task Tk is associated with   incoming task Tk , low-rank adapters {∆θkm }m∈M
a specific data distribution Dk . The objective is to      are trained on top of the base weights {θ0m }m∈M .
                                                                                    ′
sequentially acquire new knowledge from the cur-           Each update ∆θkm ∈ Rd ×d is typically formed by
rent task Tk while maintaining performance on all          the product of two low-rank matrices of rank r,
                                                       5439

## Page 4

(a) SLoRA-Pre                               (b) SLoRA-Post                                               (c) Subspace_Based Denoising
  Pretrained                                             Training Phase (Sequential)
                 Task 1 Training
   Weights
      𝜃!                   Δ𝜃#                           Task 1         Task 2           Task N
                                                                                   …                                                   Δ𝜃(
                   Subspace                             Training       Training         Training
                   Denoising
                          &#               Pretrained        Δ𝜃#            Δ𝜃$              Δ𝜃&                                             Noise
                         Δ𝜃
                                                                                                                                   𝜙'
                                            Weights                 Raw Update Buffer
                 Task 2 Training
                                               𝜃!
                                                                                                          Reference Subspace       Δ𝜃((
                           Δ𝜃$
                                                         Denoising Phase (Parallel)                            from 𝜃)*+
                   Subspace
                   Denoising                            Subspace       Subspace         Subspace
                                                        Denoising      Denoising        Denoising
                       …
                                                              &#
                                                             Δ𝜃              &$
                                                                            Δ𝜃               Δ𝜃&
                                                                                               %
                   Subspace
                   Denoising                                        Combine Updates
                        Δ𝜃&%        Final Model
                                    (𝜃! + ∑Δ𝜃%" )


Figure 2: Overview of the SLoRA framework. SLoRA mitigates catastrophic forgetting by removing noise in LoRA
updates based on subspace similarity to the base model. SLoRA-Pre performs denoising after each task, while
SLoRA-Post applies it once after all tasks. Components with low similarity to the base subspace are dropped, and
the remaining parts form the denoised update ∆θ˜ k.


typically with r ≪ min(d′ , d). Crucially, instead                          Definition 1 (Subspace-Based Adaptive Denois-
                                                                                                 ′
of directly accumulating these raw updates, they                            ing). Let ∆θk ∈ Rd ×d be the raw update for task
are processed through a subspace-based denoising                            k. Singular Value Decomposition (SVD) is per-
module. This module selectively filters compo-                              formed on the update such that ∆θk = UΣV⊤ ,
nents that are identified as noise, yielding denoised                       where U and V are orthogonal matrices contain-
updates {∆θekm }. The final parameters used for                             ing singular vectors and Σ is the diagonal matrix
inference after K tasks are composed as:                                    of singular values. Let θref be the reference weights
                                                                            with principal basis Uref . Let C ⊆ {1, . . . , r} de-
                         K
                         X                                                  note a set of candidate ranks. For any rank c ∈ C,
          m
         θK = θ0m +              ∆θeim ,     m ∈ M,                 (2)     the Grassmannian similarity ϕc is calculated as:
                           i=1
                                                                                                                               2
                                                                                                   ϕc = (Uref,c )⊤ Uc              ,                 (3)
where θKm denotes the adapted weight of module
                                                                                                                               F
m after K tasks, and ∆θeim is the denoised LoRA                             where Uc and Uref,c represent the top-c left singu-
update for each task Ti . The framework operates                            lar vectors of the update and the reference weights,
in two variants, SLoRA-Pre and SLoRA-Post, as                               respectively. The optimal rank c∗ is automatically
detailed in Section 4.4.                                                    selected by maximizing this similarity score:
4.2 Automatic Subspace-Based Denoising                                                               c∗ = arg max ϕc .                               (4)
                                                                                                                 c∈C
The core mechanism of SLoRA is the adaptive
                                                                            Consequently, the Denoised Update is recon-
denoising procedure, which filters task-specific up-
                                                                            structed using the top-c∗ singular components:
dates based on their alignment with a reference sub-
space, as shown in Figure 2(c). This approach is                                               ∆θek = Uc∗ Σc∗ (Vc∗ )⊤ ,                              (5)
motivated by the observation that the principal sin-                                                  ′    ∗
                                                                            where Uc∗ ∈ Rd ×c and Vc∗ ∈ Rd×c denote the
                                                                                                                                        ∗

gular components of pre-trained weights encapsu-                            sub-matrices containing the top-c∗ left and right
late the model’s essential capabilities (Meng et al.,                       singular vectors, respectively, and Σc∗ ∈ Rc ×c
                                                                                                                            ∗  ∗

2024). We hypothesize that update components                                is the diagonal matrix of the corresponding top-c∗
exhibiting high similarity to the base subspace rep-                        singular values.
resent meaningful adaptations that preserve general
                                                                               This subspace-based denoising removes noisy
knowledge, whereas components with low simi-
                                                                            components with low similarity to the reference
larity are likely task-specific noise or redundant
                                                                            subspace, preserving only those that are more likely
deviations that exacerbate catastrophic forgetting.
                                                                            to contribute to effective knowledge transfer across
We formally define the denoising procedure below,
                                                                            tasks. To implement this efficiently, we utilize the
omitting the module superscript m for clarity.
                                                                            randomized SVD (see Appendix C.2 for details).
                                                                       5440

## Page 5

4.3 Reference Subspace Selection                        DBpedia, and Yahoo Answers) under three differ-
A critical component of the proposed denoising          ent task orders. Detailed statistics and task orders
mechanism is the choice of the reference weight         are provided in Appendix B.1.
θref . We select the fixed pre-trained base model       Evaluation Metrics. We follow the evaluation
θ0 as the reference anchor. From a theoretical per-     metrics specified in the TRACE benchmark. In our
spective, relying on the previous model θt−1 as a       experiments, we compare the performance of each
reference introduces a risk of subspace drift, where    task individually across the sequence, as well as re-
the principal directions of new updates progres-        port the average performance across all tasks. The
sively diverge from the model’s original knowl-         task-specific evaluation metrics are consistent with
edge structure. Our analysis demonstrates that the      those defined in TRACE: accuracy for tasks like
alignment between the current update and historical     question answering and classification, ROUGE-L
subspaces decays geometrically when the reference       for summarization tasks, SARI for text simplifi-
shifts sequentially. In contrast, anchoring the de-     cation, and Edim similarity for code completion.
noising process to θ0 enforces a global consistency     For the standard CL benchmark, we evaluate the
constraint, ensuring that the principal subspaces of    accuracy and report the average accuracy over all
learned updates maintain stable alignment through-      tasks at different orders.
out the continual learning sequence. We provide
the formal definitions and detailed proofs of this      Baselines. We compare SLoRA against baselines
alignment stability analysis in Appendix A.             categorized into four groups. SLoRA belongs to
                                                        the Regularization-free category, mitigating inter-
4.4 Variants of SLoRA                                   ference by denoising the LoRA subspace without
Depending on the training setting, SLoRA operates       extra constraints. The details are as follows:
in two variants as shown in Figure 2. SLoRA-Pre
                                                        (1) Oracle methods: STL (Single-Task Learn-
(Online) performs subspace-based denoising im-
                                                            ing) trains one task in isolation and MTL
mediately after training task Tk (Figure 2(a)), inte-
                                                            (Multi-Task Learning) jointly trains on all tasks.
grating the denoised update ∆θek before proceeding
                                                            Both methods represent the upper-bound per-
to the next task. This is tailored for standard on-
                                                            formance in the continual learning setting.
line continual learning. Conversely, SLoRA-Post
(Offline) buffers raw updates from all tasks (Fig-      (2) Vanilla methods: Base uses the pretrained
ure 2(b)) and performs denoising in parallel using          model without any adaptation, and ICL per-
θ0 as the reference. This suits scenarios where raw         forms 6-shot in-context learning without pa-
checkpoints are preserved. Detailed pseudocode              rameter updates.
for both algorithms is provided in Appendix C.1.
                                                        (3) Regularization-based methods: EWC (Kirk-
5   Experiments                                             patrick et al., 2017) and LwF (Li and
                                                            Hoiem, 2017) preserve knowledge via pa-
5.1 Experimental Setup
                                                            rameter penalization and distillation, respec-
Datasets. We conduct experiments on the                     tively. GEM (Lopez-Paz and Ranzato, 2017)
TRACE benchmark (Wang et al., 2023b), which                 constrains gradients, O-LoRA (Wang et al.,
covers diverse tasks including domain-specific QA,          2023a) enforces subspace orthogonality, and
summarization, multilingual classification, and rea-        SD-LoRA (Wu et al., 2025) freezes learned
soning. To ensure a rigorous continual learning             directions while updating magnitudes.
evaluation, we follow a fixed sequential training
protocol on the eight TRACE tasks in the following      (4) Regularization-free methods: Seq-LoRA ap-
order: (1) C-STANCE, (2) FOMC, (3) Meeting-                 plies sequential LoRA fine-tuning per task as
Bank, (4) Py150, (5) ScienceQA, (6) NumGLUE-                a simple continual baseline, and RCL (Wang
cm, (7) NumGLUE-ds, and (8) 20Minuten. This                 et al., 2023b) augments training data with rea-
sequence allows us to evaluate the model’s abil-            soning traces generated by GPT-4.
ity to retain knowledge across significant domain
and task shifts. Furthermore, we evaluate on a          Implementation Details. For the TRACE bench-
standard CL benchmark consisting of five text           mark, we conduct experiments on Llama3-
classification datasets (AG News, Amazon, Yelp,         8B-Instruct, Llama3.1-8B-Instruct, Qwen2.5-7B-
                                                    5441

## Page 6

Base Model     Methods      C-STANCE   FOMC     MTB.     Py150    Sci.QA   NumG.cm   NumG.ds   20Minuten   Avg.
                STL†           56.30     68.55   61.22     70.72   93.45     70.37     73.23      41.83     66.96
                MTL†           57.90     66.94   59.72     71.10   92.30     64.20     66.15      41.50     64.98
                Base           36.15     43.55   14.97     38.66   87.15     43.21     24.62      38.26     40.82
                ICL            53.55     41.53   15.07     45.44   76.95     40.74     40.31      39.28     44.11
                EWC            41.65     52.82   33.35     29.08   74.80     25.93     39.38      40.53     42.19
                LwF            36.10     27.22    8.88     18.57   43.05     20.99     46.46      40.52     30.22
 Llama3.1-8B
                GEM            40.60     36.89   35.55     37.16   79.70     37.04     67.38      41.15     46.93
                O-LoRA         53.80     59.88   25.47     46.57   84.10     35.80     48.00      40.86     49.31
                SD-LoRA        52.75     55.65   15.81     43.54   90.35     32.10     32.31      38.49     45.13
                Seq-LoRA       43.55     53.23   26.53     43.71   87.00     54.32     72.31      41.85     52.81
                RCL            52.00     55.44   26.02     53.63   83.40     56.79     63.08      39.47     53.73
                SLoRA-Post     49.45     59.48   23.78     53.87   92.25     59.26     58.46      40.87     54.68
                SLoRA-Pre      53.80     60.48   50.80     58.23   90.40     61.73     64.92      41.19     60.19
                STL†           58.95     70.56   59.13     71.46   93.00     67.90     69.23      41.95     66.52
                MTL†           61.55     68.55   55.87     72.86   91.40     60.49     62.15      41.38     64.28
                Base           56.05     57.26   12.96     45.93   85.80     54.32     37.85      38.79     48.62
                ICL            59.30     55.65   22.63     53.46   85.80     54.32     44.00      39.70     51.86
                EWC            45.90     33.27   29.53     56.32   37.50     46.91     63.38      40.91     44.22
                LwF            44.15     44.96   29.70     53.39   67.05     48.15     59.38      40.71     48.44
 Qwen2.5-7B
                GEM            43.85     36.29   32.56     58.61   64.10     37.04     67.69      40.87     47.63
                O-LoRA         50.15     42.14   19.85     31.20   84.40     51.85     51.08      41.07     46.47
                SD-LoRA        61.30     60.48   15.37     51.25   86.45     56.79     43.38      39.11     51.77
                Seq-LoRA       51.35     55.44   24.64     43.34   75.85     56.79     63.69      41.22     51.54
                RCL            61.80     61.49   34.05     64.60   86.60     60.49     55.08      39.62     57.97
                SLoRA-Post     60.65     63.31   33.51     56.87   89.00     61.73     58.46      41.51     58.13
                SLoRA-Pre      54.25     59.68   45.96     58.72   89.65     61.73     61.85      41.72     59.20

Table 1: Experimental results on the TRACE benchmark across three foundation LLMs. ‘†’ indicates the oracle
methods. We compare SLoRA (SLoRA-Pre and SLoRA-Post) with regularization-based and regularization-free
baselines. For each task–model pair, bold and underlined represent the best and second-best results, respectively.


Instruct, Qwen2.5-14B-Instruct, and Qwen2.5-                5.2    Main Results
32B-Instruct. Regularization-based baselines (ex-
cept O-LoRA and SD-LoRA) use the official                   Table 1 reports results on the TRACE benchmark
TRACE codebase,1 while all other methods, includ-           with Llama3.1-8B, Qwen2.5-7B, and additional
ing SLoRA, are implemented within our unified               results for Llama3-8B are in Table 10. The best and
framework. To ensure a rigorous and fair compar-            second-best scores for each task are highlighted in
ison, the STL and MTL oracle baselines are also             bold and underlined, respectively. Oracle methods
implemented using LoRA with the same hyperpa-               are included for context and not ranked.
rameter configurations as our proposed SLoRA.                  Vanilla methods exhibit limited capabilities.
Unless otherwise specified, we employ LoRA with             Base performs poorly on most tasks, particularly
a rank of r=64 and a scaling factor of α=128. We            domain-specific ones like MeetingBank, while ICL
utilize the AdamW optimizer with a learning rate of         provides only marginal gains as it does not update
2 × 10−4 , a 3% linear warmup, and a cosine anneal-         parameters. Regularization-based methods gen-
ing schedule. Following TRACE, the eight tasks              erally underperform compared to Regularization-
are trained for 5/3/7/5/3/5/5/7 epochs, respec-             free approaches. On Llama3.1-8B and Llama3-8B,
tively. For the standard CL benchmark, we eval-             methods such as EWC and GEM often fall below
uate on Llama3-8B-Instruct and follow O-LoRA                the simple Seq-LoRA baseline. While SD-LoRA
for task sequences (Order1–Order3), task prompts,           shows modest gains on Qwen2.5-7B by retaining
and evaluation protocol. 2 In this setting, each task       early tasks, it degrades on later tasks, supporting
is trained for a single epoch, with all other hyper-        the observation that regularization constraints often
parameters remaining consistent. All experiments            alleviate forgetting at the cost of plasticity. Among
are conducted on four NVIDIA A100 GPUs.                     Regularization-free baselines, Seq-LoRA suffers
   1
       https://github.com/BeyonderXX/TRACE                  from substantial forgetting during sequential fine-
   2
       https://github.com/cmnfriend/O-LoRA                  tuning, while RCL improves performance but relies
                                                         5442

## Page 7

Base Model         Methods          C-STANCE            FOMC      MTB.     Py150   Sci.QA   NumG. cm   NumG. ds   20Minuten   Avg.
                     Base                 18.45           54.44     15.28    53.41   90.45      66.67     40.31       39.69     47.34
                     Seq-LoRA             48.75           59.07     47.40    69.12   91.95      59.26     50.69       42.62     58.61
 Qwen2.5-14B
                     SLoRA-Post           59.15           64.31     45.05    67.02   91.40      66.67     52.92       41.15     60.96
                     SLoRA-Pre            56.85           62.70     47.35    69.91   91.35      59.26     54.77       42.20     60.55
                     Base                 50.30           37.10     14.27    52.39   92.10      69.14     52.31       39.35     50.87
                     Seq-LoRA             51.15           58.47     34.07    23.99   95.35      76.54     70.15       42.38     56.51
 Qwen2.5-32B
                     SLoRA-Post           59.95           66.73     52.87    67.08   95.07      75.31     71.69       42.52     66.48
                     SLoRA-Pre            59.25           68.35     55.13    73.17   96.35      77.78     77.85       42.97     68.61

Table 2: Experimental results on the TRACE benchmark with Qwen2.5-14B model and Qwen2.5-32B model. Bold
and underlined denote the best and second-best results, respectively.


 Method                 Order-1     Order-2        Order-3         avg.        the gap to Oracle bounds. It maintains a balanced
 MTL†                    83.02        83.02          83.02         83.02       profile between early and later tasks, indicating re-
                                                                               duced cross-task interference without restricting
 Base                    69.29        69.29          69.29         69.29
 ICL                     58.40        60.58          63.87         60.95       learning capacity. We further verify the scalability
 EWC                     68.78        70.72          70.52         70.01       of our approach on larger architectures in Table 2.
 LwF                     69.43        64.12          68.68         67.41       SLoRA consistently outperforms the sequential
 O-LoRA                  72.59        77.22          77.24         75.68       baseline on both Qwen2.5-14B and Qwen2.5-32B.
 SD-LoRA                 72.69        72.61          72.14         72.48       Notably, on Qwen2.5-32B, SLoRA-Pre achieves
 Seq-LoRA                67.04        67.29          53.10         62.48
                                                                               an average accuracy of 68.61%, significantly sur-
 SLoRA-Post              75.83        76.29          69.19         73.77       passing Seq-LoRA (20.41%). This demonstrates
 SLoRA-Pre               76.58        78.09          73.92         76.19       that our subspace denoising mechanism remains
                                                                               robust and effective as model scale increases.
Table 3: Performance on the standard CL benchmark
with Llama3-8B, including average accuracy for Order-
                                                                               5.3   Results on Standard CL Benchmark
1, Order-2, Order-3, and overall average. Bold and
underlined denote the best and second-best results. ‘†’                        We evaluate on the standard CL benchmark
indicates the oracle methods.                                                  with Llama3-8B, as shown in Table 3. Among
                                                                               regularization-based methods, improvements over
       Method            C-STANCE      FOMC       MTB.     Py150    Avg.       Base are limited except for O-LoRA and LwF even
                           Selection Strategies                                falls below Base. The datasets in this benchmark
        min                54.30        62.30     44.15    60.43    55.30      are relatively simple and the task types are closely
       minor               54.40        62.10     50.12    64.64    57.82
     max (Ours)            55.20        65.32     50.83    64.23    58.90      aligned, which makes O-LoRA’s orthogonality con-
                           Similarity Metrics                                  straint effective in this setting and leads to the
        L2                 54.75        63.91     47.12    63.20    57.25      second-best result. SLoRA-Post performs com-
      Cosine               54.95        63.51     48.63    62.74    57.46
       PCA                 53.10        64.31     51.02    66.00    58.61      parably to O-LoRA and consistently surpasses SD-
  Frobenius (Ours)         55.20        65.32     50.83    64.23    58.90      LoRA, while SLoRA-Pre achieves the best over-
                           Reference Modules                                   all results with stronger robustness to task-order
     LoRA (last)           53.80        65.73     46.82    62.58    57.23
     LoRA (all)            53.95        65.93     45.54    63.22    57.16
                                                                               variation. These findings confirm the effective-
 Base+Prev (merged)        52.70        62.30     51.71    66.43    58.29      ness and versatility of SLoRA. We further extend
 Base+Last (separate)      55.05        65.93     47.50    59.28    56.94
 Base+Prev (separate)      54.50        62.30     47.41    63.22    56.86
                                                                               the evaluation to the 15-task long sequence follow-
    Base (Ours)            55.20        65.32     50.83    64.23    58.90      ing O-LoRA on Llama3-8B. As detailed in Ap-
                                                                               pendix B.5, SLoRA remains the top performer, and
Table 4: Ablation Study of SLoRA on first four TRACE                           SLoRA-Post shows greater stability across the long
tasks with Llama3-8B and SLoRA-Pre. We evaluate
                                                                               sequence, yielding the best final average.
different configurations of LoRA subspace selection
strategies, similarity metrics, and reference modules.
The best results in each block are highlighted in bold.
                                                                               5.4   Ablation Study
                                                                               To validate the effectiveness and impact of design
                                                                               choices in SLoRA, We conduct ablation experi-
on external data augmentation.                                                 ments on the first four TRACE tasks using Llama3-
  In contrast, SLoRA attains the strongest aver-                               8B with SLoRA-Pre. We examine three dimen-
ages across all base models, effectively narrowing                             sions: selection strategies, similarity metrics, and
                                                                            5443

## Page 8

C-STANCE          FOMC           MeetingBank                Py150                       Avg.
           Method
                        ACC↑ AFR↓        ACC↑ AFR↓        ACC↑ AFR↓               ACC↑ AFR↓                   ACC↑ AFR↓
           EWC           48.05    3.64   42.34   20.09     36.23      10.93       53.46         6.58          45.02          10.31
           GEM           44.35    4.24   40.73   12.54     36.40      12.12       54.06         9.56          43.89           9.61
           O-LoRA        46.05   12.11   51.50   20.86     31.11       8.24       54.77          3.36         45.86          11.14
           SD-LoRA       52.15    0.16   55.04   0.60      17.22      -0.68       48.53         -0.63         43.24          -0.14
           Seq-LoRA      47.95   7.06    65.73   10.85     30.90      13.06       55.54         6.26          50.03          9.31
           SLoRA-Pre     52.60   4.19    53.63   10.48     43.78        3.17      61.80          1.30         52.95           4.78
           SLoRA-Post    52.70   4.63    62.90    3.19     23.86        9.91      57.68          2.64         49.29           5.09

Table 5: Final performance (ACC) and forgetting rate (AFR) on the first four tasks in the TRACE benchmark using
Llama3-8B. ↑ means higher is better; ↓ means lower is better. SLoRA variants achieve a favorable balance of
stability and accuracy. Bold indicates the best value per column, and underline indicates the second-best.


reference modules. Table 4 reports the results and           Method
                                                             Seq-LoRA
                                                                           C-STANCE
                                                                               55.75
                                                                                             FOMC
                                                                                              70.36
                                                                                                           MeetingBank
                                                                                                               55.74
                                                                                                                               Py150
                                                                                                                                60.55
                                                                                                                                                 Avg.
                                                                                                                                                 60.60
highlights the best within each group in bold.               GEM          50.20 (-5.55)    66.94 (-3.42)    53.26 (-2.48)    58.78 (-1.77)    57.30 (-3.31)
                                                             EWC          50.20 (-5.55)    68.95 (-1.41)    50.78 (-4.96)    56.35 (-4.20)    56.57 (-4.03)

   Selection Strategies. We compare three heuris-            O-LoRA
                                                             SD-LoRA
                                                                          55.75 (+0.00)
                                                                          52.70 (-3.05)
                                                                                          59.48 (-10.88)
                                                                                          56.65 (-13.71)
                                                                                                           40.95 (-14.79)
                                                                                                           15.69 (-40.05)
                                                                                                                            57.23 (-3.32)
                                                                                                                            47.89 (-12.66)
                                                                                                                                              53.35 (-7.25)
                                                                                                                                             43.23 (-17.37)

tics for selecting singular directions from the LoRA         SLoRA-Post
                                                             SLoRA-Pre
                                                                          58.50 (+2.75)
                                                                          58.25 (+2.50)
                                                                                          64.31 (-6.05)
                                                                                          71.37 (+1.01)
                                                                                                           46.12 (-9.62)
                                                                                                           50.77 (-4.97)
                                                                                                                            60.25 (-0.30)
                                                                                                                            64.23 (+3.68)
                                                                                                                                             57.30 (-3.31)
                                                                                                                                             61.16 (+0.56)
update: min, which retains the least aligned compo-
nents; minor, which keeps intermediate ones; and            Table 6: Performance on new tasks immediately after
max (our default), which preserves the most aligned         training on Llama3-8B. Parentheses report the score
directions. Among the three, max attains the best           change vs. Seq-LoRA: positive (green) indicates im-
                                                            provement and negative (red) indicates degradation.
average performance. These results support the de-
                                                            Bold marks the best average.
noising hypothesis: base-aligned directions contain
less task noise and transfer more stably.
   Similarity Metrics. We evaluate four measures            6      Analysis
for guiding subspace selection: L2 distance, co-            6.1       Earlier Tasks Forgetting
sine similarity, PCA-based projection overlap, and
Frobenius-based subspace similarity. Frobenius-             We analyze catastrophic forgetting on Llama3-8B
based similarity achieves the highest overall aver-         using the first four TRACE tasks. Table 5 reports
age. PCA is competitive on some tasks but is less           final accuracy (ACC) and average forgetting rate
consistent across the sequence. These results indi-         (AFR; formula in Appendix B.7). Seq-LoRA at-
cate that measuring overlap at the subspace level           tains strong ACC on FOMC but shows high AFR
provides a more stable signal for denoising.                on average. Classical regularization-based meth-
                                                            ods offer limited retention at LLM scale. SD-LoRA
   Reference Modules. We compare different ref-             achieves near-zero or negative AFR yet loses ac-
erence representations used in subspace similarity          curacy on MeetingBank and Py150, which lowers
computation, including previous LoRA adapters,              its average. In contrast, both SLoRA variants of-
the base model, and their combinations. These               fer a better balance: SLoRA-Pre reaches the best
settings differ in whether similarity is computed           average ACC with a low AFR, and SLoRA-Post re-
against only the base model, specific prior tasks,          mains competitive with the second-lowest average
or merged representations, as detailed in Ap-               AFR. These results indicate that subspace denois-
pendix B.6. As shown in Table 4, the configuration          ing reduces cross-task interference while preserv-
using the base model alone consistently outper-             ing performance on earlier tasks.
forms all other variants. Alternatives involving
dynamic or merged histories fail to provide a stable        6.2       New Tasks Learning Capacity
reference frame. This validates our design choice
                                                            We examine how well models acquire new tasks
that a fixed, task-agnostic base model serves as the
                                                            when they are introduced. Table 6 reports per-
most effective anchor for subspace denoising.
                                                            formance immediately after training on each of
  Overall, these results show that each of the core         the first four TRACE tasks on Llama3-8B. Num-
components plays a crucial role in the robustness           bers in parentheses denote the change relative to
and effectiveness of the proposed SLoRA.                    Seq-LoRA. Regularization-based methods gener-
                                                         5444

## Page 9

65
                                                                                      +14.85%
                      60                                                                          +11.15%
                                                                                                              +6.58%




Average Performance
                                                                                                                         +15.82%
                      55

                      50

                      45        SLoRA-Pre (Ours)
                                SLoRA-Post (Ours)
                                Seq-LoRA
                      40
                           16          32             64       128   256
                                                    rank (r)

Figure 3: Impact of LoRA rank r on the average perfor-                        Figure 4: Universality of SLoRA across various fine-
mance across the TRACE benchmark using Qwen2.5-                               tuning methods (LoRA, DoRA, rsLoRA, and Full-FT)
7B, demonstrating that both SLoRA-Pre and SLoRA-                              on the TRACE benchmark using Qwen2.5-7B. Percent-
Post consistently achieve superior accuracy compared                          ages indicate the performance improvement of SLoRA-
to the Seq-LoRA baseline across all tested ranks.                             Pre over the Seq-LoRA baseline.


ally underperform Seq-LoRA. SD-LoRA shows                                     rsLoRA (Kalajdzievski, 2023), and Full Fine-
the weakest adaptation among regularization-based                             Tuning (Full-FT) on the TRACE benchmark using
approaches, with notable drops on MeetingBank                                 Qwen2.5-7B. As shown in Figure 4, SLoRA consis-
and Py150. Seq-LoRA adapts well to new tasks                                  tently outperforms the sequential baseline across all
but, as shown in the forgetting analysis, suffers                             methods. Notably, SLoRA-Pre yields substantial
pronounced interference later in the sequence. In                             improvements on DoRA and rsLoRA, confirming
contrast, both SLoRA variants maintain strong new-                            that noise accumulation affects various low-rank
task performance, with SLoRA-Pre achieving the                                structures. While Full-FT suffers severe forget-
best overall average and SLoRA-Post remaining                                 ting in Seq-LoRA, SLoRA-Pre effectively restores
competitive, particularly on FOMC. The pattern in-                            performance with a 15.82% improvement. These
dicates that subspace denoising preserves plasticity                          results demonstrate that subspace denoising is a
while controlling interference.                                               versatile mechanism effective for both parameter-
                                                                              efficient and full-parameter adaptation.
6.3 Sensitivity to LoRA Rank
                                                                              7   Conclusion
We analyze the impact of LoRA rank r on the av-
erage performance of the TRACE benchmark us-                                  In this paper, we are the first to identify that noise
ing Qwen2.5-7B. As illustrated in Figure 3, both                              accumulation within low-rank updates is a key fac-
SLoRA-Pre and SLoRA-Post consistently outper-                                 tor contributing to catastrophic forgetting in the
form the Seq-LoRA baseline across the entire spec-                            continual learning of LLMs. Based on this insight,
trum of ranks, ranging from r = 16 to r = 256.                                we propose SLoRA, a novel regularization-free
While we observe a performance drop at r = 256                                framework designed to mitigate cross-task inter-
across all methods, likely due to increased noise                             ference by filtering noisy components from LoRA
accumulation from the larger parameter space,                                 updates via subspace similarity analysis with the
SLoRA maintains a significant advantage over the                              frozen base model. This method is lightweight,
baseline. This consistent stability confirms that our                         requiring no rehearsal data or historical gradient
subspace denoising approach is inherently robust to                           storage while keeping the standard training pro-
variations in adapter size. Such results underscore                           cess entirely intact. It offers two versatile variants,
the practicality of SLoRA, as it achieves superior                            SLoRA-Pre and SLoRA-Post, which are specif-
performance without the necessity for fine-grained                            ically tailored for online and offline settings, re-
rank tuning.                                                                  spectively. Extensive experiments and analyses
                                                                              demonstrate that SLoRA effectively reduces forget-
6.4 Generality across Fine-tuning Methods                                     ting without compromising the model’s capacity
To assess the universality of our framework,                                  to learn new tasks, thereby achieving a balance
we extend SLoRA to DoRA (Liu et al., 2024),                                   between stability and plasticity.
                                                                           5445

## Page 10

Limitations                                              Ning Ding, Yujia Qin, Guang Yang, Fuchao Wei, Zong-
                                                           han Yang, Yusheng Su, Shengding Hu, Yulin Chen,
While SLoRA demonstrates robust performance                Chi-Min Chan, Weize Chen, and 1 others. 2023.
and scalability across various model sizes and fine-       Parameter-efficient fine-tuning of large-scale pre-
                                                           trained language models. Nature Machine Intelli-
tuning paradigms (including Full-FT and DoRA),             gence, 5(3):220–235.
we acknowledge certain limitations. First, our
subspace-based denoising mechanism relies on             Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang
                                                           Li. 2020. Orthogonal gradient descent for continual
computing similarity with the base model’s rep-            learning. In International conference on artificial
resentation space. This requires white-box access          intelligence and statistics, pages 3762–3773. PMLR.
to the pre-trained weights, limiting applicability
                                                         Xu Han, Yi Dai, Tianyu Gao, Yankai Lin, Zhiyuan Liu,
in strictly black-box API scenarios where model            Peng Li, Maosong Sun, and Jie Zhou. 2020. Contin-
parameters are inaccessible. Second, although the          ual relation learning via episodic memory activation
proposed randomized SVD approximation is highly            and reconsolidation. In Proceedings of the 58th an-
efficient, the denoising step introduces a marginal        nual meeting of the association for computational
                                                           linguistics, pages 6429–6440.
computational cost compared to naive sequential
fine-tuning. Finally, the offline variant (SLoRA-        Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski,
Post) necessitates buffering raw task updates prior        Bruna Morrone, Quentin De Laroussilhe, Andrea
                                                           Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019.
to the unified denoising phase. While these low-
                                                           Parameter-efficient transfer learning for nlp. In In-
rank updates are storage-efficient, this approach          ternational conference on machine learning, pages
incurs a temporary memory overhead proportional            2790–2799. PMLR.
to the sequence length during the training phase
                                                         Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu,
compared to purely online methods that discard             Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
updates immediately.                                       and 1 others. 2021. Lora: Low-rank adaptation of
                                                           large language models. In International Conference
Acknowledgments                                            on Learning Representations.
                                                         Joel Jang, Seonghyeon Ye, Sohee Yang, Joongbo Shin,
This work was supported by the National Natural            Janghoon Han, KIM Gyeonghun, Stanley Jungkyu
Science Foundation of China (No. 62576209) and             Choi, and Minjoon Seo. 2022. Towards continual
STCSM (No. 2025SHZDZX025G05).                              knowledge learning of language models. In Interna-
                                                           tional Conference on Learning Representations.
                                                         Shuyang Jiang, Yusheng Liao, Ya Zhang, Yanfeng
References                                                 Wang, and Yu Wang. 2024. Taia: Large language
                                                           models are out-of-distribution data learners. In The
Josh Achiam, Steven Adler, Sandhini Agarwal, Lama          Thirty-eighth Annual Conference on Neural Informa-
  Ahmad, Ilge Akkaya, Florencia Leoni Aleman,              tion Processing Systems.
  Diogo Almeida, Janko Altenschmidt, Sam Altman,
  Shyamal Anadkat, and 1 others. 2023. Gpt-4 techni-     Shuyang Jiang, Yusheng Liao, Ya Zhang, Yanfeng
  cal report. arXiv preprint arXiv:2303.08774.             Wang, and Yu Wang. 2025. Fine-tuning with re-
                                                           served majority for noise reduction. In The Thir-
Tom Brown, Benjamin Mann, Nick Ryder, Melanie              teenth International Conference on Learning Repre-
  Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind       sentations.
  Neelakantan, Pranav Shyam, Girish Sastry, Amanda
  Askell, and 1 others. 2020. Language models are        Damjan Kalajdzievski. 2023. A rank stabilization scal-
  few-shot learners. Advances in neural information        ing factor for fine-tuning with lora. arXiv preprint
  processing systems, 33:1877–1901.                        arXiv:2312.03732.
                                                         James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz,
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin,          Joel Veness, Guillaume Desjardins, Andrei A Rusu,
  Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul         Kieran Milan, John Quan, Tiago Ramalho, Ag-
  Barham, Hyung Won Chung, Charles Sutton, Sebas-          nieszka Grabska-Barwinska, and 1 others. 2017.
  tian Gehrmann, and 1 others. 2023. Palm: Scaling         Overcoming catastrophic forgetting in neural net-
  language modeling with pathways. Journal of Ma-          works. Proceedings of the national academy of sci-
  chine Learning Research, 24(240):1–113.                  ences, 114(13):3521–3526.
Bhuwan Dhingra, Jeremy R Cole, Julian Martin Eisen-      Brian Lester, Rami Al-Rfou, and Noah Constant. 2021.
  schlos, Dan Gillick, Jacob Eisenstein, and William       The power of scale for parameter-efficient prompt
  Cohen. 2022. Time-aware language models as tempo-        tuning. In Proceedings of the 2021 Conference on
  ral knowledge bases. Transactions of the Association     Empirical Methods in Natural Language Processing,
  for Computational Linguistics, 10:257–273.               pages 3045–3059.
                                                     5446

## Page 11

Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning:          David Rolnick, Arun Ahuja, Jonathan Schwarz, Timo-
  Optimizing continuous prompts for generation. In             thy Lillicrap, and Gregory Wayne. 2019. Experience
  Proceedings of the 59th Annual Meeting of the Asso-          replay for continual learning. Advances in neural
  ciation for Computational Linguistics and the 11th           information processing systems, 32.
  International Joint Conference on Natural Language
  Processing (Volume 1: Long Papers), pages 4582–            Andrei A Rusu, Neil C Rabinowitz, Guillaume Des-
  4597.                                                        jardins, Hubert Soyer, James Kirkpatrick, Koray
                                                               Kavukcuoglu, Razvan Pascanu, and Raia Hadsell.
Xilai Li, Yingbo Zhou, Tianfu Wu, Richard Socher, and          2016. Progressive neural networks. arXiv preprint
  Caiming Xiong. 2019. Learn to grow: A continual              arXiv:1606.04671.
  structure learning framework for overcoming catas-
  trophic forgetting. In International conference on         Chenyang Song, Xu Han, Zheni Zeng, Kuai Li, Chen
  machine learning, pages 3925–3934. PMLR.                     Chen, Zhiyuan Liu, Maosong Sun, and Tao Yang.
                                                               2023. Conpet: Continual parameter-efficient tun-
Zhizhong Li and Derek Hoiem. 2017. Learning without            ing for large language models. arXiv preprint
  forgetting. IEEE transactions on pattern analysis            arXiv:2309.14763.
  and machine intelligence, 40(12):2935–2947.
                                                             Hanqing Wang, Yixia Li, Shuo Wang, Guanhua Chen,
Yan-Shuo Liang and Wu-Jun Li. 2024. Inflora:                   and Yun Chen. 2024. Milora: Harnessing minor
  Interference-free low-rank adaptation for continual          singular components for parameter-efficient llm fine-
  learning. In Proceedings of the IEEE/CVF Confer-             tuning. arXiv preprint arXiv:2406.09044.
  ence on Computer Vision and Pattern Recognition,
  pages 23638–23647.                                         Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong
                                                               Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuan-Jing
Chin-Yew Lin. 2004. ROUGE: A package for automatic             Huang. 2023a. Orthogonal subspace learning for lan-
  evaluation of summaries. In Text Summarization               guage model continual learning. In Findings of the
  Branches Out, pages 74–81. Association for Compu-            Association for Computational Linguistics: EMNLP
  tational Linguistics.                                        2023, pages 10658–10671.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo              Xiao Wang, Yuansen Zhang, Tianze Chen, Songyang
  Molchanov, Yu-Chiang Frank Wang, Kwang-Ting                  Gao, Senjie Jin, Xianjun Yang, Zhiheng Xi, Rui
  Cheng, and Min-Hung Chen. 2024. Dora: Weight-                Zheng, Yicheng Zou, Tao Gui, and 1 others. 2023b.
  decomposed low-rank adaptation. In Forty-first In-           Trace: A comprehensive benchmark for continual
  ternational Conference on Machine Learning.                  learning in large language models. arXiv preprint
                                                               arXiv:2310.06762.
David Lopez-Paz and Marc’Aurelio Ranzato. 2017.
  Gradient episodic memory for continual learning. Ad-       Zifeng Wang, Zizhao Zhang, Chen-Yu Lee, Han Zhang,
  vances in neural information processing systems, 30.          Ruoxi Sun, Xiaoqi Ren, Guolong Su, Vincent Perot,
                                                                Jennifer Dy, and Tomas Pfister. 2022. Learning to
Michael McCloskey and Neal J Cohen. 1989. Catas-                prompt for continual learning. In Proceedings of
  trophic interference in connectionist networks: The           the IEEE/CVF conference on computer vision and
  sequential learning problem. In Psychology of learn-          pattern recognition, pages 139–149.
  ing and motivation, volume 24, pages 109–165. Else-
  vier.                                                      Genta Winata, Lingjue Xie, Karthik Radhakrishnan, Shi-
                                                               jie Wu, Xisen Jin, Pengxiang Cheng, Mayank Kulka-
Fanxu Meng, Zhaohui Wang, and Muhan Zhang. 2024.               rni, and Daniel Preoţiuc-Pietro. 2023. Overcoming
  Pissa: Principal singular values and singular vectors        catastrophic forgetting in massively multilingual con-
  adaptation of large language models. Advances in             tinual learning. In Findings of the Association for
  Neural Information Processing Systems, 37:121038–            Computational Linguistics: ACL 2023, pages 768–
  121072.                                                      777.
Fuli Qiao and Mehrdad Mahdavi. 2024. Learn more,             Tongtong Wu, Linhao Luo, Yuan-Fang Li, Shirui Pan,
  but bother less: parameter efficient continual learning.     Thuy-Trang Vu, and Gholamreza Haffari. 2024a.
  Advances in Neural Information Processing Systems,           Continual learning for large language models: A sur-
  37:97476–97498.                                              vey. arXiv preprint arXiv:2402.01364.
Roger Ratcliff. 1990. Connectionist models of recog-         Yichen Wu, Hongming Piao, Long-Kai Huang, Renzhen
  nition memory: constraints imposed by learning               Wang, Wanhua Li, Hanspeter Pfister, Deyu Meng,
  and forgetting functions. Psychological review,              Kede Ma, and Ying Wei. 2025. Sd-lora: Scalable
  97(2):285.                                                   decoupled low-rank adaptation for class incremental
                                                               learning. In The Thirteenth International Conference
Anastasia Razdaibiedina, Yuning Mao, Rui Hou, Ma-              on Learning Representations.
  dian Khabsa, Mike Lewis, and Amjad Almahairi.
  2023. Progressive prompts: Continual learning for          Yichen Wu, Hong Wang, Peilin Zhao, Yefeng Zheng,
  language models. In The Eleventh International Con-          Ying Wei, and Long-Kai Huang. 2024b. Mitigating
  ference on Learning Representations.                         catastrophic forgetting in online continual learning
                                                         5447

## Page 12

by modeling previous task interrelations via pareto     Assumption 1 (Subspace Evolution Hypothesis).
    optimization. In Forty-first International Conference   We assume that the principal basis Ut of the de-
    on Machine Learning.
                                                            noised update for task t is derived from the ref-
Wei Xu, Courtney Napoles, Ellie Pavlick, Quanze Chen,       erence basis Uref perturbed by a task-specific in-
  and Chris Callison-Burch. 2016. Optimizing statis-        novation. Formally, Ut can be represented as a
  tical machine translation for text simplification. In     linear combination of the reference basis and an
 Transactions of the Association for Computational
  Linguistics, volume 4, pages 401–415.                     orthogonal innovation basis Nt :

Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju                       Ut = Uref Rt + Nt St ,            (6)
   Hwang. 2018. Lifelong learning with dynamically          where Rt and St are coefficient matrices. This
   expandable networks. In International Conference
   on Learning Representations.                             decomposition adheres to the following properties:
                                                               1. Subspace Alignment: The denoising process
Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel.
  2022. Bitfit: Simple parameter-efficient fine-tuning            ensures that Ut maintains significant align-
  for transformer-based masked language-models. In                ment with the reference. This is quantified by
  Proceedings of the 60th Annual Meeting of the As-               the Frobenius norm of their product:
  sociation for Computational Linguistics (Volume 2:
                                                                                             2
  Short Papers), pages 1–9.                                                    (Uref )⊤ Ut       = γt ,     (7)
                                                                                             F
Friedemann Zenke, Ben Poole, and Surya Ganguli.
   2017. Continual learning through synaptic intelli-             where 0 < γt < rank(Ut ) represents the
   gence. In International conference on machine learn-           magnitude of retained information.
   ing, pages 3987–3995. PMLR.
                                                               2. Orthogonality of Innovation: The innovation
Juzheng Zhang, Jiacheng You, Ashwinee Panda, and                  basis Nt captures new task knowledge that
  Tom Goldstein. 2025. Lori: Reducing cross-task                  is orthogonal to the reference subspace, i.e.,
  interference in multi-task low-rank adaptation. arXiv
  preprint arXiv:2504.07448.                                      (Uref )⊤ Nt = 0.

Qingru Zhang, Minshuo Chen, Alexander Bukharin,                3. High-Dimensional Sparsity: Due to the high
  Pengcheng He, Yu Cheng, Weizhu Chen, and                        dimensionality of the parameter space, task-
  Tuo Zhao. 2023. Adaptive budget allocation for                  specific innovations from different tasks are
  parameter-efficient fine-tuning. In The Eleventh In-            mutually nearly orthogonal. For t ̸= i:
  ternational Conference on Learning Representations.
                                                                                             2
Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015.                             (Nt )⊤ Ni        ≈ 0.       (8)
                                                                                             F
  Character-level convolutional networks for text clas-
  sification. In Advances in Neural Information Pro-           To evaluate the consistency of the learned sub-
  cessing Systems, pages 649–657.                           spaces across the lifespan of the model, we define
Ming Zhong, Chenxin An, Weizhu Chen, Jiawei Han,            the following metric:
  and Pengcheng He. 2024. Seeking neural nuggets:           Definition 2 (Historical Alignment Score). The
  Knowledge transfer in large language models from a
                                                            Historical Alignment Score A(t) measures the av-
  parametric perspective. In The Twelfth International
  Conference on Learning Representations.                   erage subspace consistency between the current
                                                            task update and historical updates. Consistent with
A     Theoretical Proofs                                    our denoising metric, it is defined as the Frobenius
                                                            norm of the product of their principal bases:
This appendix provides the mathematical deriva-
                                                                                    t−1
tion for the reference subspace selection analysis                             1 X          2
discussed in Section 4.3. We rigorously define the                   A(t) =        (Ut )⊤ Ui ,              (9)
                                                                              t−1           F
                                                                                    i=1
alignment metric and analyze its asymptotic behav-
ior under two different reference strategies: using         where Ut and Ui represent the matrix of top-c∗ left
the previous model (θt−1 ) versus using the fixed           singular vectors (principal basis) of the denoised
base model (θ0 ).                                           updates ∆θet and ∆θei , respectively.
                                                               A high A(t) indicates that the principal direc-
A.1 Definitions and Assumptions                             tions of the current update remain aligned with the
We start by formalizing the relationship between            subspaces established by previous tasks. Based on
the reference subspace and the learned update sub-          this, we present the following theorem to validate
space.                                                      our choice of θ0 .
                                                        5448

## Page 13

A.2 Proof of Lemma 1                                                       A.3      Proof of Theorem 1
To prove the main theorem, we introduce a lemma                            Theorem 1 (Alignment Stability Analysis). Let
describing how subspace alignment decays when                              A(prev) (t) denote the alignment score when us-
the reference frame shifts sequentially.                                   ing the previous model θt−1 as the reference, and
Lemma 1 (Geometric Decay of Subspace Consis-                               A(base) (t) denote the score when using the base
tency). Let the reference basis for task t be the prin-                    model θ0 . As the number of tasks t → ∞:
cipal basis of the previous task, i.e., Uref = Ut−1 .
The subspace alignment between the current basis                              lim A(prev) (t) = 0,           lim A(base) (t) = C, (17)
                                                                              t→∞                         t→∞
Ut and a historical basis Ui (i < t) satisfies the
recurrence relation:                                                       where C is a positive constant.
                     √                                                     Proof. The proof analyzes the asymptotic behav-
    (Ut )⊤ Ui      ≤ γt · (Ut−1 )⊤ Ui . (10)
                    F                                     F                ior of  the subspace consistency metric A(t) =
                                                                             1 Pt−1          ⊤
                                                                                   i=1 ∥(Ut ) Ui ∥F .
                                                                                                   2
Consequently, the alignment score decays geomet-                           t−1
rically with the distance t − i:                                              Case 1: θref = θt−1
                                              t                               Here, the reference basis for task t is Ut−1 . Ac-
                            2                 Y
          (Ut )⊤ Ui             ≤ Cstart ·           γk ,          (11)    cording to Lemma 1, the alignment between Ut
                            F
                                             k=i+1                         and any historical basis Ui decays as the distance
                                                                           t − i increases. Let γmax = supk γk < 1 (normal-
where γk < 1 (normalized) acts as the retention
                                                                           ized retention factor). We bound the term:
factor.
                                                                                                         2
Proof. Substituting Assumption 1 with Uref =                                             (Ut )⊤ Ui                    t−i
                                                                                                              ≤ C0 · γmax .         (18)
                                                                                                         F
Ut−1 , we express Ut as:
                                                                           Substitute this into the alignment score:
               Ut = Ut−1 Rt + Nt St .                              (12)
                                                                                                    t−1                       t−1
We examine the interaction between Ut and the                                                  C0 X t−i    C0 X j
                                                                              A(prev) (t) ≤        γmax =      γmax .
distant historical basis Ui . Computing the matrix                                            t−1         t−1
                                                                                                    i=1                       j=1
product:                                                                                                                    (19)
     ⊤                  ⊤           ⊤
(Ut ) Ui = (Rt ) (Ut−1 ) Ui +(St ) (Nt ) Ui .       ⊤             ⊤        As t → ∞, the sum of the geometric series con-
                                                                                                    γmax                      1
                                               (13)                        verges to a constant 1−γ    max
                                                                                                           , but the factor t−1
Based on the High-Dimensional Sparsity property                            drives the total value to 0:
(Assumption 1), the current innovation Nt is or-
thogonal to the historical basis Ui (which is com-                                             lim A(prev) (t) = 0.                 (20)
                                                                                              t→∞
posed of previous innovations and references), so
∥(Nt )⊤ Ui ∥F ≈ 0. The expression simplifies to:                            Case 2: θref = θ0
                                                                            In this case, the reference basis for all tasks is
    (Ut )⊤ Ui           ≈ (Rt )⊤ (Ut−1 )⊤ Ui                  . (14)       U0 . According to Assumption 1, for any task k,
                F                                         F
                                                                           Uk is derived from U0 :
Using the sub-multiplicative property of matrix
norms (∥AB∥F ≤ ∥A∥2 ∥B∥F ) and noting that                                                Uk = U0 Rk + Nk Sk .                      (21)
∥Rt ∥2 ≈ ∥Rt ∥F for low-rank matrices in this con-
text:                                                                      We evaluate the alignment term ∥(Ut )⊤ Ui ∥2F :

   (Ut )⊤ Ui        ≤ ∥Rt ∥F · (Ut−1 )⊤ Ui                        . (15)         (Ut )⊤ Ui = (Rt )⊤ (U0 )⊤ U0 Ri + . . .
                F                                             F
                                                                                                                                    (22)
                                                      2                                       ≈ (Rt )⊤ Ri .
From Assumption 1, (Ut−1             = γt .  )⊤ U   t F
Since Ut−1 has orthonormal columns,                                        Note that the cross terms involving Nt and Ni
                         √
 (Ut−1 )⊤ Ut F = ∥Rt ∥F = γt . Thus:                                       vanish due to orthogonality with U0 and mutual
                            √                                              orthogonality (Assumption 1). Thus, the alignment
    (Ut )⊤ Ui           ≤       γt · (Ut−1 )⊤ Ui              .    (16)
                    F                                     F                depends on the retained signal matrices Rt and Ri .
Recursively applying this relation yields the geo-                                                   2                    2
metric decay.                                                                          (Ut )⊤ Ui         ≈ (Rt )⊤ Ri           .    (23)
                                                                                                     F                    F
                                                                       5449

## Page 14

Dataset             Source       Avg Len       Metric        Language   #Data
                                                                                       Dataset            Task               Domain
Domain-specific
ScienceQA           Science        210       Accuracy         English   5000
FOMC                Finance         51       Accuracy         English   5000
                                                                                       Yelp        sentiment analysis      Yelp reviews
MeetingBank         Meeting       2853       ROUGE-L          English   5000           Amazon      sentiment analysis     Amazon reviews
Multi-lingual                                                                          DBpedia     topic classification     Wikipedia
C-STANCE          Social media     127       Accuracy        Chinese    5000
20Minuten            News          382        SARI           German     5000           Yahoo       topic classification    Yahoo Q&A
Code completion                                                                        AG News     topic classification       news
Py150               Github         422     Edim similarity    Python    5000
Mathematical reasoning
NumGLUE-cm         Math            32        Accuracy         English   5000       Table 8: The details of datasets used in the standard CL
NumGLUE-ds         Math            21        Accuracy         English   5000       benchmark.
Table 7: Overview of dataset statistics in the TRACE
benchmark.                                                                             Order                Task Sequence
                                                                                         1       dbpedia → amazon → yahoo → ag
Assuming a consistent learning stability where                                           2       dbpedia → amazon → ag → yahoo
the magnitude of information retained from                                               3       yahoo → amazon → ag → dbpedia
the base subspace is stable across tasks (i.e.,
E[∥(Rt )⊤ Ri ∥2F ] = C > 0), the average score                                     Table 9: Three different task sequences of the standard
                                                                                   CL benchmark used for continual learning experiments.
becomes:
                                                   t−1
                                       1 X                                         on the standard CL benchmark for language mod-
    lim A(base) (t) = lim                  C = C. (24)
    t→∞                          t→∞ t − 1                                         els. This benchmark contains five text classifica-
                                                   i=1
                                                                                   tion datasets drawn from different domains (Zhang
This confirms that anchoring to θ0 maintains a glob-                               et al., 2015): AG News, Amazon reviews, Yelp
ally consistent principal subspace.                                                reviews, DBpedia and Yahoo Answers. Table 8
                                                                                   summarizes the task types and domains. Follow-
B       Experiment Details                                                         ing O-LoRA, we report results under three task
                                                                                   sequences (Table 9) to evaluate robustness against
B.1       Dataset Statistcs
                                                                                   task-order variation. Unlike TRACE, this bench-
TRACE To evaluate the robustness and gener-                                        mark focuses on homogeneous text classification
alizability of our method across diverse tasks and                                 tasks, making it a complementary setting for as-
modalities, we conduct experiments on the TRACE                                    sessing continual learning methods.
benchmark, which consists of eight datasets span-                                     All datasets do not contain personally identifi-
ning multiple domains, languages, and task types.                                  able information or offensive content
As summarized in Table 7, the benchmark includes
domain-specific tasks such as question answer-                                     B.2       Additional Implementation Details
ing (ScienceQA), financial document classification                                 The open-source code, models, and datasets uti-
(FOMC), and meeting summarization (Meeting-                                        lized in our experiments adhere to their respective
Bank); multilingual tasks in Chinese and German                                    licenses and are used in accordance with their in-
(C-STANCE and 20Minuten); code completion in                                       tended purposes.
Python (Py150); and mathematical reasoning tasks
(NumGLUE-cm and NumGLUE-ds). Each dataset                                          B.3       Pre-Experiments
contains 5,000 samples and is evaluated using ap-                                  We design a controlled two-task experiment to val-
propriate metrics such as accuracy, ROUGE-L (Lin,                                  idate the effect of noisy components in LoRA up-
2004), SARI (Xu et al., 2016), or embedding simi-                                  dates. Following the standard CL benchmark, we
larity. 3 This diverse composition enables a compre-                               select Amazon Reviews as Task1 (sentiment analy-
hensive assessment of continual learning methods                                   sis) and Yahoo Answers as Task2 (topic classifica-
under heterogeneous task settings, linguistic varia-                               tion), which differ in task type. We train Llama3-
tion, and input distributions.                                                     8B sequentially on Task1 and Task2 with standard
                                                                                   Seq-LoRA.
standard CL benchmark For comparison with
                                                                                      After training, we proportionally drop LoRA pa-
prior continual learning work, we also evaluate
                                                                                   rameters under three conditions: DropFirst (operate
    3
        https://github.com/seatgeek/fuzzywuzzy                                     on the Task1 adapter only), DropSecond (operate
                                                                                5450

## Page 15

Base Model     Methods           C-STANCE            FOMC     MTB.     Py150        Sci.QA        NumG. cm      NumG. ds         20Minuten       Avg.
                STL†                  55.50           70.77    56.76     68.79          94.10        60.49             73.85         41.82        65.26
                MTL†                  55.95           67.54    55.23     70.20          92.60        66.67             66.46         41.65        64.54
                Base                  44.05           53.83    14.38     42.04          80.10        32.10             19.69         37.26        40.43
                ICL                   46.85           58.27    13.30     44.81          84.70        40.74             25.85         38.79        44.16
                EWC                   48.05           42.34    36.23     53.46          60.95        27.16             64.00         40.78        46.62
                LwF                   40.90           38.31     6.56     26.81          40.15        12.35             53.23         39.50        32.23
 Llama3-8B
                GEM                   44.35           40.73    36.40     54.06          63.50        28.40             65.54         40.69        46.71
                O-LoRA                51.50           50.20    31.11     54.77          81.50        60.49             58.15         40.50        53.53
                SD-LoRA               52.15           55.04    17.22     48.53          87.00        33.34             30.46         39.32        45.38
                Seq-LoRA              47.95           65.73    30.90     55.54          82.00        49.38             71.38         41.11        55.50
                RCL                   58.20           50.40    14.14     63.40          84.40        61.73             53.54         39.79        53.20
                SLoRA-Post            52.70           62.90    23.86     57.68          90.00        66.67             63.69         40.75        57.28
                SLoRA-Pre             52.60           53.63    43.78     61.80          90.65        59.26             67.38         41.47        58.82

Table 10: Experimental results on the TRACE benchmark in Llama3-8B. ‘†’ indicates the oracle methods excluded
from ranking. We compare SLoRA (SLoRA-Pre and SLoRA-Post) with regularization-based and regularization-free
baselines. For each task–model pair, bold and underlined denote the best and second-best continual methods.

Methods       MNLI    CB      WIC     COPA    QQP     BoolQA   RTE     IMDB      Yelp    Amazon     SST-2    DBpedia     AG.    MultiRC   Yahoo    Avg.
Seq-LoRA      50.49   60.71   50.78   87.00   76.92    72.42   59.93   95.75    46.08      42.22    94.15     87.28     83.68    82.20    74.01   70.91
O-LoRA        43.65   58.93   53.76   91.00   68.07    67.98   54.87   95.29    63.87      57.09    92.09     93.76     83.92    77.91    70.34   71.50
SD-LoRA       58.40   75.00   52.35   90.00   78.70    78.10   57.76   95.24    63.08      57.89    92.09     87.66     84.51    75.27    69.50   74.37
SLoRA-Pre     42.80   60.71   50.16   85.00   70.57    77.25   53.07   95.59    66.28      61.95    93.69     92.34     87.37    73.78    71.72   72.15
SLoRA-Post    64.47   75.00   52.35   92.00   77.82    79.82   64.26   95.01    63.26      56.37    91.40     82.88     85.38    79.43    71.36   75.35


Table 11: Performance comparison of Seq-LoRA, O-LoRA, SD-LoRA, SLoRA-Pre and SLoRA-Post based on the
Llama3-8B model on long-sequence benchmark. The task order follows Order-4 in the O-LoRA paper.


on the Task2 adapter only), and DropAll (operate                               → copa → qqp → boolqa → rte → imdb → yelp
on both). The drop ratio ranges from 10% to 90%.                               → amazon → sst-2 → dbpedia → ag → multirc
Dropping is carried out by removing the less im-                               → yahoo. We conduct experiments on Llama3-8B
portant components in the LoRA update subspace                                 across all 15 tasks.
in proportion to the specified drop ratio.                                        Table 11 compares Seq-LoRA, O-LoRA, SD-
                                                                               LoRA, and our two variants of SLoRA. Seq-LoRA
B.4   Main Results on Llama3-8B                                                attains strong results on individual datasets such as
Table 10 reports results on the TRACE bench-                                   IMDB and SST-2 but suffers from severe degrada-
mark using Llama3-8B. Regularization-based ap-                                 tion on others, leading to a lower average. O-LoRA
proaches remain weak at this scale, often falling be-                          improves stability on tasks like WIC and DBpedia
low Seq-LoRA. Seq-LoRA itself achieves strong re-                              yet remains behind in overall performance. SD-
sults on some tasks such as FOMC and NumGLUE-                                  LoRA achieves suboptimal average performance
ds but suffers from pronounced forgetting, while                               due to its better performance in early tasks. SLoRA-
RCL excels on C-STANCE yet lacks consistency.                                  Pre shows notable gains on sentiment and topic
In contrast, SLoRA-Pre attains the best average                                classification datasets such as Yelp, Amazon, and
and SLoRA-Post is second-best among contin-                                    AG News, while SLoRA-Post achieves the high-
ual learners. SLoRA-Pre leads on MeetingBank,                                  est average by maintaining strong and balanced
Py150, and ScienceQA, while SLoRA-Post per-                                    results across the entire sequence. These results
forms best on NumGLUE-cm, confirming that sub-                                 confirm that SLoRA remains effective under long
space denoising improves retention without sacri-                              sequences, and that the post-hoc variant offers su-
ficing adaptation.                                                             perior robustness in this setting.

B.5   Experiments on the Long-Sequence                                         B.6      Reference Modules for Subspace
                                                                                        Similarity
To further evaluate continual learning under
more challenging conditions, we follow the long-                               In our ablation study on reference modules (Ta-
sequence setting introduced in O-LoRA and adopt                                ble 4), we evaluated six configurations to calculate
the same task order (Order-4): mnli → cb → wic                                 subspace similarity during the denoising process.
                                                                       5451

## Page 16

These configurations determine which prior repre-                Method                         A(2)                  A(3)             A(4)
                                                                 Seq-LoRA                       1.7792                1.7750          1.7101
sentations serve as the reference when measuring                 LoRA (last)               1.9592 (+10.12%)    1.8837 (+6.12%)    1.8276 (+6.87%)
the alignment between the current LoRA update                    LoRA (all)                1.5770 (-11.36%)    1.5581 (-12.22%)   1.5182 (-11.22%)
                                                                 Base+Prev (merged)        1.9874 (+11.70%)    1.8807 (+5.95%)    1.8581 (+8.65%)
and historical model behavior. Specifically, we                  Base+Last (Separate)      1.5224 (-14.43%)    1.5304 (-13.78%)   1.5164 (-11.33%)
                                                                 Base+Prev (Separate)      1.5309 (-13.96%)    1.5500 (-12.68%)   1.4756 (-13.71%)
compare the following designs:
                                                                 Base (Ours)               2.0026 (+12.56%)    1.8966 (+6.85%)    1.8693 (+9.31%)


   • LoRA (last): The similarity is calculated with       Table 12: Historical Alignment Score A(t) (Higher is
     the LoRA adapter from the immediately pre-           Better) for different reference selection strategies on the
     ceding task Tt−1 .                                   first four TRACE tasks. Percentage values in parenthe-
                                                          ses show the improvement relative to the Seq-LoRA.
   • LoRA (all): The similarity is calculated with
                                                                            1.00                                                    Seq-LoRA
     the union of all previous LoRA adapters
                                                                            0.98                                                    SLoRA-Post
     {∆θ1 , . . . , ∆θt−1 }.



                                                        Cosine Similarity
                                                                            0.96
   • Base+Prev (merged): All previous LoRA                                  0.94
     adapters
          P are merged into the base model:                                 0.92
     θ0 + t−1i=1 ∆θi . The current update is then                           0.90
     compared to this merged model, as an approx-
                                                                            0.88
     imation of the full historical subspace.
                                                                                   1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31
   • Base+Last (separate): The similarity is com-                                                             Layer
     puted separately with both θ0 and ∆θt−1 , and         Figure 5: Cosine similarity between hidden states on
     the scores are averaged.                              the first task (C-STANCE), measured both immediately
                                                           after task-specific training and after completing the full
   • Base+Prev (separate): The similarity is com-          task sequence.
     puted separately with θ0 and each prior ∆θi
     for i < t, and the final similarity score is
                                                           B.7                     Average Forgetting Rate Analysis on
     averaged between them.
                                                                                   Early Tasks
   • Base (Ours): Only the base model θ0 is used          To better quantify catastrophic forgetting under
     as the reference, consistent with our default        continual learning, we compute the Average For-
     design in SLoRA.                                     getting Rate (AFR) for all methods on the first
                                                          four tasks of the TRACE benchmark—C-STANCE,
   All variants use the same underlying similar-          FOMC, MeetingBank, and Py150—using the
ity metric (Grassmannian similarity via Frobenius         LLaMA3-8B model. For a given task Tk , let Rk (i)
norm). The difference lies solely in the reference        denote the model’s performance on Tk after train-
subspace, which impacts the stability and general-        ing up to task Ti . AFR is defined as:
ity of the denoising process.                                                        K
                                                                              1    X
   To further validate the theoretical findings from        AFR(Tk ) =                  (Rk (k) − Rk (i)) ,
Appendix B, particularly Theorem 1 which sup-                              K −k
                                                                                  i=k+1
ports anchoring to the base model θ0 to maintain                                                         (25)
consistent alignment, we compute the Historical           where K is the total number of tasks. AFR mea-
Alignment Score A(t) (defined in Appendix A,              sures the average degradation in performance on
Definition 2) for the variants above. The results in      Tk after subsequent tasks are learned, lower values
Table 12 show that explicitly anchoring to the fixed      indicate better retention.
base model (Base (Ours)) yields the highest A(t)
across tasks t = 2, 3, 4, demonstrating superior his-      B.8                     Representation Drift in Hidden States
torical subspace consistency compared to methods          As shown in Figure 5, the cosine similarity tends
that rely on accumulated or previous task updates.        to decrease with layer depth for both methods, sug-
This numerical evidence supports the core design          gesting that deeper layers experience greater repre-
choice of selecting θ0 as the fixed reference anchor      sentational changes during continual learning. No-
for denoising.                                            tably, SLoRA-Post consistently maintains higher
                                                    5452

## Page 17

8 0.37 1.00 0.28 0.89 0.64 0.68 0.97   1.0         8 0.54 0.98 0.27 0.90 0.66 0.69 0.96   1.0                         65         Seq-LoRA               SLoRA-Post (Ours)
       7 0.39 0.82 0.31 0.83 0.72 0.74 0.98   0.8         7 0.35 0.76 0.34 0.83 0.71 0.77 0.99   0.8                                    SLoRA-Simple           SLoRA-Pre (Ours)




                                                                                                       Average Performance
       6 0.36 0.93 0.29 0.94 0.70 0.77 0.98               6 0.42 0.93 0.32 0.93 0.75 0.88 0.97                               60
                                              0.6                                                0.6
 Task ID                                            Task ID
       5 0.41 0.92 0.29 0.97 0.79 0.78 0.99               5 0.49 0.97 0.31 0.97 0.74 0.81 0.99
       4 0.37 0.91 0.30 0.93 0.66 0.66 0.97               4 0.35 0.88 0.27 0.94 0.68 0.74 0.97                               55
                                              0.4                                                0.4
       3 0.42 0.97 0.22 0.94 0.70 0.70 0.97               3 0.50 1.00 0.32 0.97 0.67 0.74 0.98                               50
       2 0.42 0.93 0.30 0.90 0.74 0.72 0.97   0.2         2 0.37 0.95 0.27 0.90 0.78 0.73 0.99   0.2
       1 0.46 1.00 0.31 0.93 0.72 0.74 0.91               1 0.37 1.00 0.33 0.92 0.73 0.68 0.91                               45
                                              0.0                                                0.0
         roj roj roj roj roj roj roj                        roj roj roj roj roj roj roj
      q_p k_p v_p o_p up_gpate_pown_p                    q_p k_p v_p o_p up_gpate_pown_p                                     40
                                d                                                  d                                              Llama3-8B      Llama3.1-8B          Qwen2.5-7B
                      Module                                             Module                                                                        Model
                         (a)                                                (b)                                                                   (c)

Figure 6: Module retention and effects of denoising strategies. (a) and (b) show module-wise retention ratios after
denoising on Llama3-8B for SLoRA-Pre and SLoRA-Post. (c) Average performance across three base models
comparing SLoRA-Pre and SLoRA-Post with SLoRA-Simple, which removes q_proj and v_proj at inference.


similarity than Seq-LoRA, demonstrating that our                                          Algorithm 1 SLoRA-Pre: Online Subspace-Based
denoising strategy is effective in preserving task-                                       Denoising
specific representations while mitigating represen-                                       Require: Full base model weights θ0 , LoRA tar-
tation drift.                                                                                 get modules M, task sequence {T1 , . . . , TK },
                                                                                              candidate ranks C
B.9        Module Retention after Denoising                                                1: for each m ∈ M do
                                                                                           2:   Compute SVD: θ0m = Um          m    m ⊤
We analyze module-wise parameter retention af-                                                                              0 Σ0 (V0 )
ter denoising on Llama3-8B for SLoRA-Pre and                                               3: end for
SLoRA-Post. Figures 6(a) and 6(b) report aver-                                             4: for k = 1 to K do
age retention ratios across tasks. A consistent                                            5:   for each m ∈ M do
pattern emerges: q_proj and v_proj retain the                                              6:      Train LoRA update ∆θkm on task Tk over
                                                                                                     m
fewest parameters, up_proj and gate_proj show                                                      θk−1
moderate retention, while k_proj, o_proj, and                                              7:      Compute SVD: ∆θkm = Um Σm (Vm )⊤
down_proj are largely preserved. Motivated by the                                          8:      for each c ∈ C do
low retention in q_proj and v_proj, we introduce                                           9:         Compute similarity:
a simple variant, SLoRA-Simple, which removes                                                         ϕm           m ⊤ m 2
                                                                                                        c = (U0,c ) Uc F
these two LoRA modules at inference. Figure 6(c)                                          10:      end for
shows that SLoRA-Simple brings marginal gains                                             11:      c∗ ← arg maxc∈C ϕm    c
on Llama3-8B but reduces stability on Llama3.1-                                           12:      Reconstruct
8B and Qwen2.5-7B. In contrast, SLoRA-Pre                                                           f m = Um∗ Σm∗ (Vm∗ )⊤
                                                                                                   ∆θ  k        c   c     c
achieves the highest overall average, indicating that                                     13:      θkm ← θk−1
                                                                                                            m + ∆θ  fm
                                                                                                                       k
similarity-driven filtering is more effective and ro-                                     14:   end for
bust than fixed module removal.                                                           15: end for
   To understand how continual learning affects                                           16: return Final weights {θK   m } where

internal model representations, we measure the co-                                                      (         P      fm
sine similarity between hidden states on the first                                                m       θ0m + K   k=1 ∆θ k , if m ∈ M
task (C-STANCE) at two points: immediately af-                                                   θK =
                                                                                                          θ0m ,                 otherwise
ter training on the task itself and after completing
the full sequence of continual learning tasks. This
similarity reflects how much the model’s internal
representations shift over time, with lower values                                        C      Implementation and Algorithmic
indicating greater drift. We conduct this analysis                                               Details
on the Llama3-8B model using 100 randomly se-
lected examples from the C-STANCE test set. For                                           C.1     Algorithm: Subspace-Based Denoising in
each example, we compute the cosine similarity                                                    SLoRA
between hidden states from the two checkpoints                                            We present the full pseudocode for both variants
across all 32 transformer layers.                                                         of SLoRA. Algorithm 1 describes SLoRA-Pre,
                                                                                   5453

## Page 18

Algorithm 2 SLoRA-Post: Offline Subspace-                 We follow a standard randomized SVD pipeline
Based Denoising                                         with orthogonalization and projection:
Require: Full base model weights θ0 , LoRA tar-             1. Random projection. Sample a Gaussian
    get modules M, task sequence {T1 , . . . , TK },           random matrix P ∈ Rd×c and compute the
    candidate ranks C                                                                       ′
                                                               sketch matrix Y = ∆θkm P ∈ Rd ×c .
 1: for each m ∈ M do
 2:   Compute SVD: θ0m = Um          m    m ⊤
                                 0 Σ0 (V0 )
                                                            2. Orthonormal basis. Perform QR decompo-
                                                                                            ′
 3: end for                                                    sition Y = QR, where Q ∈ Rd ×c has or-
 4: for k = 1 to K do                                          thonormal columns.
 5:   for each m ∈ M do
                                                            3. Low-dimensional projection. Project the
 6:      Train LoRA update ∆θkm on task Tk over
                                                               original matrix into the subspace spanned by
         θ0m
                                                               Q:
 7:      Store ∆θkm
                                                                         B = Q⊤ ∆θkm ∈ Rc×d .          (26)
 8:   end for
 9: end for                                                 4. SVD in projected space. Compute SVD of
10: for k = 1 to K do                                          B:
11:   for each m ∈ M do                                                     B = ÛΣV⊤ ,          (27)
12:      Compute SVD: ∆θkm = Um Σm (Vm )⊤
                                                                 where Û ∈ Rc×c , Σ ∈ Rc×c , and V ∈ Rc×d .
13:      for each c ∈ C do
14:         Compute similarity:                             5. Approximate SVD. Recover the approximate
            ϕm           m ⊤ m 2
              c = (U0,c ) Uc F                                 left singular vectors in the original space:
15:      end for                                                                                   ′

16:      c∗ ← arg maxc∈C ϕm                                                         U = QÛ ∈ Rd ×c .         (28)
                              c
17:      Reconstruct
          f m = Um∗ Σm∗ (Vm∗ )⊤                             We then construct the rank-c approximation:
         ∆θ  k        c   c    c
18:   end for                                                        ∆θk
                                                                         m(c)
                                                                                    = Uc · diag(Σc ) · Vc ,   (29)
19: end for
                                                                                ′
20: for each m ∈ M do                                   where Uc ∈ Rd ×c and Vc ∈ Rc×d denote the
                      PK f m
21:   θKm ← θm +
               0        k=1 ∆θ k
                                                        top-c singular directions, and diag(Σc ) forms a
22: end for                                             diagonal matrix from the top-c singular values.
23: return Final weights {θK  m } where                   To ensure reliable similarity measurement in the
                                                                                                      m(c)
              (         P                               denoising stage, we apply a final SVD to ∆θk
                θ0m + K       fm
        m                 k=1 ∆θ k , if m ∈ M           and retain the top-r left singular vectors as the
       θK =
                θ0m ,                 otherwise         subspace basis for alignment scoring.
                                                        Comparison with Standard SVD. The full SVD
                                                        of a d′ × d matrix costs O(d′ × d2 ) time. In con-
which applies subspace-based denoising immedi-          trast, the randomized SVD reduces the complex-
ately after each task and incrementally accumulates     ity to O(d × r2 ), assuming r ≪ d′ , by operating
denoised LoRA updates. Algorithm 2 describes            in a lower-dimensional space. Our experiments
SLoRA-Post, which performs standard LoRA train-         show this approximation preserves denoising qual-
ing across all tasks and applies denoising once after   ity while significantly reducing computational over-
the full sequence is completed.                         head, enabling scalable continual learning with
                                                        LLMs.
C.2 Efficient Computation of Low-Rank SVD
                                                        D       AI Assistance Statement
To improve efficiency, we approximate the singu-
lar value decomposition (SVD) of each LoRA up-          Language editing and stylistic refinement of the
                 ′
date ∆θkm ∈ Rd ×d using a randomized projec-            draft were performed with the aid of large lan-
tion method. As each update is formed via low-          guage models, including ChatGPT 4 and Google
rank decomposition (i.e., ∆θkm = Bkm Am k with          Gemini 5 .
r ≪ min(d′ , d)), computing a full-rank SVD is              4
                                                                https://chatgpt.com/
unnecessary.                                                5
                                                                https://gemini.google.com/app/

                                                    5454
