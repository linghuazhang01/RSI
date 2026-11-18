# Extracted pages: 2605.29295.pdf

## Page 1

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization


                                               Tao Jiang 1 2 Xinmeng Yu 1 2 Chenhao Yi 3 2 Yiling Wu 2 Yan Li 2 Ran Cheng 4 5 6 Dongmei Jiang 2 †
                                                                                     Jianguo Zhang 1 2 7 †

                                                                                                                                                 MMLU
                                                                  Abstract
                                                                                                                                                   0.6
                                                                                                                                AbstainQA                           MMLU-Pro
                                              Evolutionary model merging provides a power-                                                         0.4




arXiv:2605.29295v1 [cs.NE] 28 May 2026
                                              ful framework for the automated, training-free
                                              composition of LLMs through parameter-space                                                          0.2


                                              search. However, existing methods predomi-
                                                                                                                      TruthQA                                               HellaSwag
                                              nantly rely on stochastic, hand-crafted operators
                                              that overlook the underlying performance land-
                                              scape of the coefficient space. We propose Evo-
                                              lutionary Generative Merging (EvoGM), a frame-                               NLGraph                               K-Cross
                                              work that transcends manual heuristics by em-
                                              ploying learnable generative modeling to opti-                                                     GSM8K
                                              mize merging coefficients. Specifically, EvoGM
                                              features a dual-generator architecture with cycle-                                   base
                                                                                                                                   single best
                                                                                                                                                   pso-merging
                                                                                                                                                   model swarm
                                                                                                                                                                     ours


                                              consistent learning to adaptively sample and re-
                                              fine promising merging candidates. By construct-                  Figure 1. Single-task performance comparison of 10 fine-tuned
                                              ing winner-loser pairs from historical search tra-                Qwen2.5-1.5B models. The proposed method outperforms base-
                                                                                                                line models on almost all targeted tasks.
                                              jectories, our framework effectively captures high-
                                              performance parameter distributions and maxi-
                                              mizes data efficiency. This generative process
                                              is seamlessly integrated into a multi-round evo-
                                                                                                                1. Introduction
                                              lutionary pipeline, where elite merged models                     The prevailing paradigm of large language models (LLMs)
                                              iteratively serve as new expert foundations. Ex-                  relies on large-scale pretraining followed by task-specific
                                              tensive experiments across diverse benchmarks                     adaptation, yet the rapid growth in model size makes full-
                                              demonstrate that EvoGM significantly outper-                      parameter fine-tuning increasingly impractical under realis-
                                              forms state-of-the-art baselines, exhibiting robust               tic computational and data constraints (Hu et al., 2022; Li
                                              performance on both seen and unseen tasks. Code                   & Liang, 2021; Lv et al., 2024). In this context, a key chal-
                                              and data are available at https://github.                         lenge is how to accumulate and compose model capabilities
                                              com/JiangTao97/evogm.                                             without repeatedly training new models. Model merging
                                                                                                                addresses this challenge by directly integrating multiple ex-
                                                                                                                pert models into a single one, enabling efficient knowledge
                                            †                           1
                                              Corresponding authors.      Research Institute of Trustwor-       transfer and capability composition at no additional training
                                         thy Autonomous Systems and Department of Computer Sci-                 cost, and has therefore emerged as a promising mechanism
                                         ence and Engineering, Southern University of Science and               for scalable and sustainable model improvement (Lu et al.,
                                         Technology 2 Pengcheng Laboratory 3 University of Chinese
                                         Academy of Sciences 4 Department of Data Science and Artifi-           2024a; Yang et al., 2026; Zheng et al., 2025).
                                         cial Intelligence, The Hong Kong Polytechnic University 5 Hong         Most existing model merging methods rely on manually
                                         Kong Polytechnic University Shenzhen Research Institute 6 Hong
                                         Kong Polytechnic University-Daya Bay Technology and Inno-              designed heuristics, including linear averaging (Ilharco et al.,
                                         vation Research Institute 7 Guangdong Provincial Key Labo-             2023; Jin et al., 2025; Tang et al., 2024; Wortsman et al.,
                                         ratory of Brain-inspired Intelligent Computation. Correspon-           2022; Yu et al., 2024a), spherical interpolation (Jiang et al.,
                                         dence to: Dongmei Jiang <jiangdm@pcl.ac.cn>, Jianguo Zhang             2025), or the use of scaling and sparsifying schemes to
                                         <zhangjg@sustech.edu.cn>.                                              mitigate parameter interference (Ma et al., 2025; Sun et al.,
                                         Proceedings of the 43 rd International Conference on Machine           2025b; Yang et al., 2025). These approaches are largely task-
                                         Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026           agnostic and lack the ability to adapt their merging strategies
                                         by the author(s).                                                      according to validation objectives. To address this limitation,

                                                                                                            1

## Page 2

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization




Figure 2. Overview of EvoGM. We optimize merging coefficients λ ∈ RN for task-vector merging. (1) Population Initialization:
initialize a diverse population of λ (average merge, one-hot, random) and evaluate on Dval to build the history set H = {(λ, f (λ))}.
(2) Winner–Loser Pairing: split H into winners H+ and losers H− . (3) Dual-Generator Training: train (G−→+ , G+→− ) with cycle-
consistency and optimization-guided losses. (4) Generative Evolution & Selection: apply G−→+ to generate new candidates and select
improved λ by evaluation. (5) Evolving Expert Basis: periodically refresh the expert pool via elite merges and recompute {τi }N  i=1 ,
returning the best λ∗ and merged model θ(λ∗ ).


more recent studies have explored optimization- or search-             structured exploration of the merging space and improved
based formulations (Li et al., 2025a; Liu et al., 2025; Wang           generalization.
et al., 2020). Furthermore, evolutionary model merging
                                                                       Building on this insight, we propose Evolutionary Gen-
has emerged as a practical alternative, enabling gradient-
                                                                       erative Merging (EvoGM), a unified framework that inte-
free population-based search and demonstrating competitive
                                                                       grates generative modeling into evolutionary model merging.
performance across a wide range of tasks (Akiba et al., 2025;
                                                                       EvoGM formulates the search over merging coefficients
Inoshita et al., 2024; Mencattini et al., 2025).
                                                                       as adaptive sampling from a learned probability distribu-
Despite their empirical success, existing evolutionary model           tion, replacing static, hand-crafted mutation operators. The
merging methods largely operate over an unstructured                   framework employs a dual-generator architecture with cycle-
search space and depend on stochastic perturbations, re-               consistent constraints to capture the geometric structure of
sulting in low search efficiency under realistic deployment            high-performance regions while preserving diversity. To
constraints with small validation sets and limited evaluation          effectively leverage past search experience, we introduce
budgets. In optimization problems characterized by simi-               a winner-loser preference mechanism that constructs com-
larly sparse and expensive feedback, generative optimization           parative signals from historical trajectories, enabling the
approaches have been shown to substantially reduce search              generative model to distinguish promising directions from
randomness by learning structured proposal distributions               ineffective ones. Embedded within a multi-round evolu-
from limited evaluative signals (Gao et al., 2024; Jiang et al.,       tionary pipeline, elite merged models iteratively serve as
2026; Wang et al., 2025). Motivated by these advances, we              new expert foundations, allowing the search strategy and
argue that model merging should be formulated as a data-               model capabilities to co-evolve. Extensive experiments
driven and learnable search process rather than a purely               demonstrate that EvoGM achieves significantly improved
stochastic one. By treating validation performance as a                efficiency and robustness, consistently outperforming prior
reward signal, a generative model can capture the distribu-            approaches on both seen and unseen tasks.
tion of high-performing merging configurations, enabling
                                                                       Our main contributions are summarized as follows:

                                                                   2

## Page 3

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

   • We reformulate evolutionary model merging as a learn-          Algorithm 1 EvoGM: Evolutionary Generative Merging
     able generative task, enabling adaptive characterization         Input: Base model θpre , initial experts {θi }Ni=1 , valida-
     of high-performance regions.                                     tion set Dval , population size P , number of rounds R,
                                                                      number of iterations per round T
   • We introduce a dual-generator architecture with cycle-
                                                                      Output: Optimal merging coefficients λ∗
     consistency and a winner-loser preference strategy to
                                                                      Compute initial task vectors: τi ← θi − θpre
     maximize data utilization from sparse feedback, en-
                                                                      for round r = 1 to R do
     abling efficient model synthesis for unseen tasks.
                                                                         P ← InitPop(P ); F ← f (θ(P), Dval )
   • Extensive experiments demonstrate that EvoGM sig-                   H ← {(λ(p) , F (p) )}Pp=1

     nificantly outperforms state-of-the-art baselines across            Reset Dual-Generators G−→+ , G+→−
     diverse benchmarks and model families.                              for iteration t = 1 to T do
                                                                            H+ , H− ← PairWinnersLosers(H)
                                                                            Train Dual-Generators G−→+ , G+→−
2. Methodology                                                              λnew ← G−→+ (H+ )
                                                                            Fnew ← f (θ(λnew ), Dval )
2.1. Problem Formulation
                                                                            H ← H ∪ {(λnew , Fnew )}; P ← Select(H, P )
We formulate model merging as an optimization task that                  end for
seeks the optimal coefficients for combining task vectors.               if r < R then
Let θpre ∈ Rd denote the parameters of a pre-trained base                   Select top-N coefficients {λ(i) }N
                                                                                                             i=1 from history H
model, and {θi }N i=1 be a set of N expert models fine-tuned                Update experts: θi ← θ(λ(i) )
from θpre for specific tasks. Following the principles of                   Re-compute task vectors: τi ← θi − θpre
task arithmetic, we define the task vector for each expert               end if
as τi = θi − θpre . The merged model is parameterized                 end for
by a coefficient vector λ = [λ1 , . . . , λN ]⊤ ∈ RN , where          return best θ(λ∗ ) found in H
the dimensionality of the optimization space corresponds
                              PN models N . This results in
directly to the number of expert
parameters θ(λ) = θpre + i=1 λi τi . Given a validation             Population Initialization The search process begins by
set Dval and an evaluation metric f (·), our objective is to        initializing a population P = {λ(p) }P  p=1 of candidate con-
find the optimal coefficients λ∗ = arg maxλ f (θ(λ); Dval ).        figurations. Throughout this work, we set the population
In this framework, the optimization is restricted to the low-       size to P = 2N . To ensure diverse coverage of the co-
dimensional vector λ ∈ RN .                                         efficient space, we employ a hybrid initialization strategy
                                                                    comprising: (i) average merging, where λi = 1/N for all
Limitations Finding λ∗ is non-trivial. Grid search is com-          i; (ii) one-hot vectors, representing individual experts; and
putationally prohibitive due to the exponential growth of           (iii) random sampling from a uniform distribution. Each
the search space with N , while existing evolutionary meth-         candidate λ(p) is evaluated on Dval to compute its perfor-
ods rely on stochastic perturbations and manual heuristics.         mance score f (θ(λ(p) ), Dval ). These initial evaluations
These approaches often fail to account for the underlying           establish the baseline performance and provide the essential
performance distribution, frequently leading to sub-optimal         seed trajectories for generative learning.
results in complex merging scenarios.
                                                                    Winner-Loser Pairing In each iteration, we maintain a
                                                                    search history archive H = {(λ, f (λ))} that stores all pre-
      Can we transcend manual search heuristics by
                                                                    viously evaluated coefficient configurations and their fitness
    learning to locate optimal merging configurations?
                                                                    scores. We split H into a winner set H+ and a loser set H−
                                                                    using a ratio ρ (default 0.3), where H+ contains the top-ρ
2.2. Evolutionary Generative Merging                                fraction of configurations. For clarity, we denote samples
                                                                    drawn from the loser and winner sets as λ− ∈ H− and
To address the challenge, we propose to transform the search        λ+ ∈ H+ , respectively.
process into a learnable generative task. Instead of directly
                                                                    We construct a paired training set via the Cartesian product:
perturbing λ through random mutations, EvoGM charac-
terizes the distribution of optimal merging coefficients by             H− × H+ = {(λ− , λ+ ) | λ− ∈ H− , λ+ ∈ H+ }.
modeling the relationship between historical search trajec-
tories and their corresponding performance on Dval . The            We form mini-batches by independently sampling λ− ∼
EvoGM framework is illustrated in Figure 2, with the full           H− and λ+ ∼ H+ , which is equivalent to sampling from
procedure detailed in Algorithm 1.                                  the product distribution.

                                                                3

## Page 4

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

Train Dual-Generator To model transitions between dif-              distribution toward regions with higher performance po-
ferent performance levels in the coefficient space RN , we          tential. These new candidates are subsequently evaluated
employ a dual-generator architecture consisting of a for-           on Dval to obtain their fitness Fnew . We then update the
ward generator G−→+ (mapping losers to winners) and a               search history H ← H ∪ {(λnew , Fnew )} and refine the
backward generator G+→− (mapping winners to losers).                population P by selecting the top-P individuals based on
Both generators are implemented as five-layer MLPs, with            their fitness scores. This process ensures that the population
a tanh activation in the output layer to constrain the gener-       continuously migrates toward higher-performance regions
ated coefficients λnew within the range [−1, 1].                    within the coefficient space.

   • Training Objective. The generators are jointly opti-           Evolving Expert Basis A distinctive feature of EvoGM
     mized by balancing cycle-consistency and optimization          is that the expert pool itself evolves across outer rounds,
     guidance:                                                      which we refer to as a basis shift. While the inner loop
                                                                    optimizes the merging coefficients λ over a fixed set of task
                  Ltotal = αc Lcyc + αo Lopt ,                      vectors, the outer loop explicitly redefines the underlying
                                                                    expert basis by updating the task vectors {τi }. At the end
     where αc and αo control the contributions of cycle-            of each round (r < R), we select the top-N coefficient
     consistency regularization and optimization guidance,          vectors {λ(i) }N i=1 from the accumulated history archive H.
     respectively.                                                  These elite configurations are then used to synthesize a new
                                                                                                         (r+1)            
   • Cycle-Consistency Loss. To ensure that the learned             expert pool for the next round: θi          = θ λ(i) , i =
     mapping is reversible and preserves the structural dis-        1, . . . , N, where θ(λ(i) ) denotes the merged model con-
     tribution of the coefficient population, we impose the         structed by applying λ(i) to the experts of round r. We
     following cycle-consistency constraint:                        subsequently recompute task vectors with respect to the
                                                                                                (r+1)     (r+1)
                       h                                i           pretrained base model: τi         = θi      − θpre . This basis
                                                      2
     Lcyc = Eλ− ∈H− ∥G+→− (G−→+ (λ− )) − λ− ∥2                      shift enables EvoGM to progressively refine the expert di-
                         h                                 i        rections available for merging, rather than searching over a
                                                         2
             + Eλ+ ∈H+ ∥G−→+ (G+→− (λ+ )) − λ+ ∥2 .                 static expert pool.

     Minimizing Lcyc regularizes G−→+ to capture the ge-            3. Experimental Setup
     ometric relationship between H− and H+ and prevents
     trivial mode collapse.                                         We study model merging under two settings: merging mod-
                                                                    els trained on seen tasks and merging models evaluated on
   • Optimization-Guided Loss. Beyond consistency, we               unseen tasks. Here, seen tasks refer to tasks on which the
     explicitly guide the generative search toward high-            expert models are explicitly fine-tuned, while unseen tasks
     performance regions. Let the centroid of the winner            denote tasks that are not used during expert fine-tuning. Ex-
     set be                                                         periments span two model architectures, FLAN-T5 (Chung
                            1    X
                   µ+ =      +
                                       λ+ .                         et al., 2024) and Qwen (Bai et al., 2023), different merging
                          |H | + +
                                 λ ∈H                               scales (8 and 10 models), and both single-task and multi-task
     The optimization loss is defined as                            configurations.
                            h                i
                                           2
          Lopt = Eλ− ∈H− ∥G−→+ (λ− ) − µ+ ∥2 .                      Model Configuration For the seen tasks setting, we use
                                                                    FLAN-T5-base1 together with 8 expert models fine-tuned
     This objective drives G−→+ to transform sub-optimal            on single corresponding tasks. For the unseen tasks set-
     configurations toward the statistical average of elite         ting, we construct a set of ten domain-specific experts2 by
     solutions, thereby accelerating the discovery of the           independently fine-tuning Qwen2.5-1.5B 3 on each of the
     global optimum λ∗ .                                            ten supervised fine-tuning (SFT) domains in Tulu-v24 (Ivi-
                                                                    son et al., 2023). All experts are adapted using LoRA-
                                                                    based parameter-efficient fine-tuning (Hu et al., 2022). Each
Generative Evolution and Selection With the dual-
                                                                    model is trained for five epochs with an initial learning rate
generators optimized, the forward generator G−→+ serves
as a learned evolutionary operator. In each iteration t,               1
                                                                         https://huggingface.co/google/flan-t5-base
                                                                       2
we generate a set of new candidates by applying the                      https://huggingface.co/TaoJiangCN/qwen2.5-1.
                                                                    5b-tulu-v2-lora-experts
learned mapping to the current population P: λnew =                    3
                                                                         https://huggingface.co/Qwen/Qwen2.5-1.5B
G−→+ (λ), ∀λ ∈ P. By transforming the entire popu-                     4
                                                                         https://huggingface.co/collections/allenai/
lation through G−→+ , the framework shifts the candidate            tulu-v2-suite


                                                                4

## Page 5

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

Table 1. Single-task performance comparison of 10 fine-tuned Qwen2.5-1.5B models. Best results are highlighted with bold and light-gray
background, and second-best results are underlined, computed independently for each column (Val/Test).

                       MMLU          MMLU-Pro HellaSwag           K-Cross          GSM8K              NLGraph           TruthQA           AbstainQA
        Method
                      Val     Test   Val   Test   Val    Test    Val     Test     Val     Test        Val      Test     Val     Test      Val        Test
          Base       0.570 0.564 0.271 0.207 0.590 0.581 0.410 0.297 0.255 0.155 0.285 0.284 0.460 0.425 0.185 0.101
          MTL        0.620 0.566 0.286 0.207 0.525 0.484 0.395 0.327 0.365 0.188 0.320 0.319 0.360 0.305 0.005 0.003
       Single Best   0.615 0.586 0.386 0.232 0.620 0.572 0.410 0.392 0.395 0.325 0.355 0.376 0.495 0.384 0.210 0.119
 Model Soup (22)     0.610 0.574 0.300 0.225 0.590 0.578 0.355 0.330 0.300 0.397 0.285 0.197 0.465 0.283 0.070 0.048
    TA (23)          0.620 0.570 0.286 0.234 0.605 0.588 0.370 0.391 0.350 0.260 0.280 0.286 0.465 0.389 0.075 0.060
   TIES (23)         0.600 0.557 0.329 0.221 0.510 0.470 0.345 0.342 0.295 0.302 0.390 0.316 0.495 0.416 0.075 0.028
  DARE (23)          0.605 0.576 0.300 0.233 0.615 0.593 0.380 0.390 0.320 0.243 0.285 0.286 0.455 0.395 0.095 0.043
   CMA (25)      0.615 0.576 0.386 0.217 0.600 0.592 0.415 0.392 0.430 0.343 0.430 0.445 0.530 0.428 0.170 0.075
PSO-Merging (25) 0.635 0.595 0.386 0.232 0.650 0.587 0.430 0.393 0.490 0.354 0.455 0.465 0.525 0.421 0.260 0.147
Model Swarm (25) 0.640 0.606 0.371 0.236 0.635 0.587 0.410 0.392 0.410 0.332 0.360 0.373 0.495 0.392 0.230 0.095
          Ours       0.640 0.625 0.429 0.224 0.660 0.594 0.430 0.379 0.495 0.434 0.545 0.537 0.540 0.441 0.230 0.121



of 2 × 10−4 and an effective batch size of 32.                         2023; Zhang et al., 2024); and safety-related evaluation
                                                                       is conducted on TruthfulQA (Lin et al., 2021), and Ab-
Baselines For the seen tasks setting, we compare against               stainQA (Gehman et al., 2020). Unless otherwise specified,
a broad range of representative merging approaches, includ-            we randomly sample 200 examples for validation and 1,000
ing TA (Ilharco et al., 2023), DARE (Yu et al., 2024b),                examples for testing on each dataset.
TIES (Yadav et al., 2023), DARE-TIES (Yadav et al., 2023;
Yu et al., 2024b), DELLA-Merging (Deep et al., 2024),                  3.1. Seen Tasks
RankMean (Perin et al., 2024), CMA (Akiba et al., 2025),               We evaluate the performance of merging 8 FLAN-T5 models
AdaMerging (Yang et al., 2024), Fisher-Merging (Matena                 on their corresponding tasks. Baseline results are taken from
& Raffel, 2022), and RegMean (Jin et al., 2023). For the               PSO-Merging for comparison. The multi-task performance
unseen tasks setting, we consider both training-based and              on the GLUE benchmark is summarized in Table 2. Our
merging-based baselines. These include the base pretrained             proposed method consistently achieves the highest average
model (Base), multi-task learning (MTL), the best single ex-           accuracy among all tested approaches, providing a relative
pert (Single Best), and Model Soup (Wortsman et al., 2022),            improvement of over 1.4% compared to the previous state-
as well as merging methods such as TA, TIES, DARE, CMA,                of-the-art, PSO-Merging. In semantic similarity (STS-B),
PSO-Merging (Zhang et al., 2025), and Model Swarm (Feng                our approach yields a relative performance gain of more
et al., 2025).                                                         than 12% over PSO-Merging and approximately 6% over
                                                                       the best-performing traditional merging baseline.
Benchmarks and Tasks For the seen tasks setting, we
evaluate model merging on a suite of eight text-to-text gen-
                                                                       Table 2. Performance comparison on 8 GLUE benchmark tasks.
eration tasks drawn from the GLUE benchmark (Wang                      Best results are highlighted with bold and light-gray background,
et al., 2018), covering grammaticality judgment, natural               and second-best results are underlined.
language inference, paraphrase identification, sentiment
analysis, and semantic similarity. The tasks include CoLA,                Method        CoLA MNLI MRPC QNLI QQP RTE SST-2 STS-B AVG

MNLI, MRPC, QNLI, QQP, RTE, SST-2, and STS-B 5 .                            TA          69.1   62.7     79.4     89.8   83.9   81.2    91.7   73.2    78.9
                                                                           DARE         69.5   63.8     79.7     89.9   83.9   81.2    91.7   69.8    78.7
For the unseen tasks setting, we assess generalization on                  TIES         69.2   59.4     77.7     89.3   83.4   80.5    91.3   68.4    77.4
                                                                        DARE-TIES       69.3   62.5     79.7     89.8   83.8   81.6    91.3   71.1    78.6
eight datasets spanning three categories: knowledge, rea-                 DELLA         69.3   64.4     79.9     89.9   83.8   82.0    91.1   76.0    79.5
soning, and safety. The knowledge category includes                      RankMean       69.1   56.5     76.2     88.5   82.1   80.1    91.2   62.2    75.7
                                                                           CMA          70.9   82.9     75.7     89.4   73.9   80.9    92.2   69.7    79.4
MMLU (Hendrycks et al., 2021), MMLU-Pro (Wang et al.,                   AdaMerging      69.9   77.2     79.9     89.8   81.7   79.1    91.4   66.1    79.4
2024), and HellaSwag (Zellers et al., 2019); reasoning tasks               Fisher       69.3   54.0     76.7     84.6   83.6   77.6    88.1   74.4    76.0
                                                                          RegMean       69.1   26.6     75.3     79.3   77.2   61.7    86.0   48.1    65.4
cover GSM8K (Cobbe et al., 2021), Knowledge Cross-                      PSO-Merging     68.2   83.8     80.6     89.5   83.6   81.2    91.1   71.9    81.2
words (Ding et al., 2024), and NLGraph (Wang et al.,
                                                                           Ours         71.1   82.8     78.9     88.1   84.8 82.2      90.0   80.9    82.4
   5
    https://huggingface.co/collections/tanganke/
flan-t5-base-models-fine-tuned-on-glue-benchmark


                                                                  5

## Page 6

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

Table 3. Multi-task performance comparison of 10 fine-tuned Qwen2.5-1.5B models. Best results are highlighted with bold and light-gray
background, and second-best results are underlined, computed independently for each column (Val/Test).

                         MMLU        MMLU-Pro HellaSwag        K-Cross          GSM8K        NLGraph      TruthQA      AbstainQA
       Method                                                                                                                       Average
                        Val   Test   Val   Test   Val   Test   Val       Test   Val   Test   Val   Test   Val   Test   Val   Test
        Base            0.570 0.564 0.271 0.207 0.590 0.581 0.410 0.297 0.255 0.155 0.285 0.284 0.460 0.425 0.185 0.101              0.352
        MTL             0.620 0.566 0.286 0.207 0.525 0.484 0.395 0.327 0.365 0.188 0.320 0.319 0.360 0.305 0.005 0.003              0.330
     Single Best        0.615 0.586 0.386 0.232 0.620 0.572 0.410 0.392 0.395 0.325 0.355 0.376 0.495 0.384 0.210 0.119              0.405
   Model Soup (22)      0.610 0.574 0.300 0.225 0.590 0.578 0.355 0.330 0.300 0.397 0.285 0.197 0.465 0.283 0.070 0.048              0.350
 Task Arithmetic (23)   0.620 0.570 0.286 0.234 0.605 0.588 0.370 0.391 0.350 0.260 0.280 0.286 0.465 0.389 0.075 0.060              0.364
      TIES (23)         0.600 0.557 0.329 0.221 0.510 0.470 0.345 0.342 0.295 0.302 0.390 0.316 0.495 0.416 0.075 0.028              0.356
     DARE (23)          0.605 0.576 0.300 0.233 0.615 0.597 0.380 0.390 0.320 0.243 0.285 0.286 0.455 0.395 0.095 0.043              0.364
     CMA (25)           0.600 0.569 0.357 0.217 0.645 0.609 0.405 0.415 0.275 0.211 0.275 0.275 0.500 0.421 0.140 0.078              0.375
  PSO-Merging (25)      0.580 0.539 0.314 0.197 0.530 0.532 0.415 0.414 0.315 0.261 0.380 0.352 0.520 0.418 0.130 0.059              0.372
  Model Swarm (25)      0.610 0.573 0.343 0.229 0.585 0.545 0.390 0.411 0.375 0.246 0.305 0.297 0.440 0.410 0.130 0.071              0.372
        Ours            0.620 0.576 0.314 0.225 0.635 0.588 0.405 0.388 0.350 0.248 0.295 0.290 0.455 0.397 0.165 0.130              0.380



3.2. Unseen Tasks                                                         prove results through weight search, their efficacy is often
                                                                          constrained by the stochasticity of their operators. Con-
Merging on unseen tasks effectively benchmarks general-
                                                                          versely, EvoGM navigates the high-performance parame-
ization under data scarcity. By optimizing weights on small
                                                                          ter space more effectively through its generative architec-
validation sets, evolutionary methods support both single-
                                                                          ture. This advantage is especially pronounced in AbstainQA,
task specialization (tailoring weights to specific task) and
                                                                          where EvoGM exhibits a 66% relative improvement over
multi-task merging (identifying a unified model for all tasks
                                                                          the strongest evolutionary baseline. These results demon-
simultaneously).
                                                                          strate that by leveraging historical search trajectories, our
                                                                          framework discovers balanced weight combinations that
Single-task We evaluate the merging of 10 Qwen2.5-                        successfully mitigate task conflicts in few-shot scenarios.
1.5B experts across 8 unseen tasks by optimizing dedi-
cated weight configurations for each task independently.
                                                                          4. Analysis
As shown in Table 1 and Figure 1, EvoGM achieves the
highest test accuracy on 5 of 8 benchmarks. For instance,                 Convergence and Optimization Efficiency We analyze
a 15% relative performance gain is observed in NLGraph                    the convergence of EvoGM by comparing it with Model
compared to PSO-Merging. Notably, multi-task learning                     Swarm and PSO-Merging. All methods are evaluated using
and traditional merging methods like TIES or DARE often                   a population size of 20. To visualize the optimization pro-
fail to outperform the single best expert. This indicates that            cess, we plot the average performance of the top 5 solutions
static merging schemes or joint training struggle to transfer             from each iteration.
specialized knowledge to new domains. In contrast, evolu-
tionary methods such as CMA and PSO-Merging achieve                       As shown in Figure 3, EvoGM consistently outperforms the
better results by searching for optimal weights on validation             baselines across all tasks. The baseline algorithms typically
sets. EvoGM further advances these baselines by incorpo-                  reach a performance plateau around iteration 5, showing
rating learnable generative modeling, achieving superior                  little improvement thereafter. In contrast, EvoGM follows
and more robust performance across complex reasoning and                  a multi-round optimization strategy, configured here with
knowledge tasks.                                                          2 rounds and 3 iterations per round. A noticeable perfor-
                                                                          mance shift occurs at iteration 3, which marks the transition
                                                                          between rounds. At this stage, the framework performs a
Multi-task We evaluate the generalization of a single                     basis shift by generating new parameter candidates based
merged model across 8 unseen tasks simultaneously. As                     on the elite models from the previous round. While this
shown in Table 3, EvoGM achieves a leading average score                  re-initialization causes a temporary fluctuation, it effectively
of 0.380 among all merging approaches. Notably, standard                  prevents the search from getting trapped in local optima.
multi-task fine-tuning yields a score of 0.330, which falls               This phase allows EvoGM to explore more promising re-
below the base model performance of 0.352. This perfor-                   gions of the parameter space, leading to substantial perfor-
mance drop underscores the severe inter-task interference                 mance gains in the subsequent round.
inherent in joint fine-tuning across diverse domains. While
evolutionary baselines such as CMA and PSO-Merging im-

                                                                     6

## Page 7

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

                                          MMLU-Pro                                                            TruthfulQA                                                       MMLU                                                        NLGraph
                                                                                           0.54                                                         0.70
                                                                                                                                                                  Val                                                          Val
                      0.42
                                                                                                                                                                  Test                                                         Test
                                                                                                                                                                                                                     0.55




Top-5 Mean Accuracy                                                  Top-5 Mean Accuracy
                      0.40                                                                                                                              0.65
                                                                                           0.52
                                                                                                                                                                                                                     0.50
                      0.38

                                                                                                                                             Accuracy                                                     Accuracy
                                                                                                                                                        0.60
                                                                                           0.50                                                                                                                      0.45
                      0.36
                                                                                                                                                        0.55                                                         0.40
                      0.34
                                                                                           0.48
                                                       EvoGM                                                               EvoGM                        0.50                                                         0.35
                      0.32
                                                       PSO-Merging                                                         PSO-Merging
                      0.30                             Model Swarm                         0.46                            Model Swarm                                                                               0.30
                                                                                                                                                               Single    w/o Rounds   w/o Cycle   EvoGM                     Single    w/o Rounds   w/o Cycle   EvoGM
                             0        2          4          6                                     0       2          4          6
                                           Iteration                                                           Iteration
                                                                                                                                                                        (a) MMLU                                                (b) NLGraph
                                 (a) MMLU-Pro                                                         (b) TruthfulQA                                                          TruthQA                                                     HellaSwag
                                                                                                                                                                  Val                                                          Val
                                                                                                                                                                                                                     0.70
                                          NLGraph                                                             HellaSwag                                 0.55
                                                                                                                                                                  Test                                                         Test
                      0.55
                                  EvoGM                                                    0.66
                                                                                                                                                                                                                     0.65
                                  PSO-Merging




Top-5 Mean Accuracy                                                  Top-5 Mean Accuracy
                                                                                                                                                        0.50


                                                                                                                                             Accuracy                                                     Accuracy
                      0.50        Model Swarm
                                                                                           0.64
                                                                                                                                                                                                                     0.60
                                                                                                                                                        0.45
                      0.45
                                                                                           0.62                                                                                                                      0.55
                                                                                                                                                        0.40
                      0.40
                                                                                           0.60                                                                                                                      0.50
                                                                                                                           EvoGM                        0.35
                      0.35
                                                                                                                                                               Single    w/o Rounds   w/o Cycle   EvoGM                     Single    w/o Rounds   w/o Cycle   EvoGM
                                                                                           0.58                            PSO-Merging
                                                                                                                           Model Swarm
                      0.30
                             0        2          4          6                                     0       2          4          6
                                                                                                                                                                    (c) TruthQA                                                (d) HellaSwag
                                           Iteration                                                           Iteration
                                                                                                                                             Figure 4. Ablation study on different components. (1) Single-
                                  (c) NLGraph                                                         (d) HellaSwag                          Generator represents the single generative model variant; (2) w/o
Figure 3. Fitness evolution of EvoGM and SOTA methods in multi-                                                                              Rounds represents the replacement of multi-round updates with
task scenarios. The curves represent the mean performance of the                                                                             five continuous iterations; and (3) w/o Cycle Loss represents the
top five individuals in the population, with shaded areas indicating                                                                         removal of the cycle-consistency constraint.
the confidence intervals computed from these individuals.

                                                                                                                                             and reporting the average performance across 2 tasks. As
Ablation Study We conduct an ablation study to eval-                                                                                         summarized in Table 12 of the Appendix, the population
uate the contribution of each core component in EvoGM.                                                                                       size exhibits a significant influence on search quality. In-
This experiment involves merging 4 expert models with                                                                                        creasing the population from 10 to 30 leads to a 0.0305
a population size of 8. The full framework is configured                                                                                     gain in test accuracy. This suggests that a larger number
with 2 rounds and 2 iterations per round. We compare the                                                                                     of candidates allows for a more thorough exploration of
full version against 3 variants: (1) Single-Generator, which                                                                                 the high-dimensional parameter manifold. In contrast, the
uses only a single generative model; (2) w/o Rounds, which                                                                                   number of iterations per round has a relatively stable effect,
replaces the multi-round expert update with 5 continuous                                                                                     with a slight performance increase observed as iterations
iterations on the initial experts; and (3) w/o Cycle Loss,                                                                                   scale from 3 to 8.
which removes the cycle-consistency constraint. Results are
summarized in Figure 4 and Table 11 in Appendix.                                                                                             Regarding the training of the generative model, we find that
                                                                                                                                             200 to 400 epochs are sufficient. Increasing the epochs to
Each component is shown to be critical for achieving opti-                                                                                   600 causes a minor performance drop of 0.0035, likely due
mal performance across diverse benchmarks. The removal                                                                                       to the generator over-fitting on specific historical search
of the multi-round mechanism (w/o Rounds) leads to a no-                                                                                     trajectories. For the learning rate, a smaller value of 0.0001
ticeable decline in complex tasks such as NLGraph and                                                                                        yields the best results, providing a 0.0220 improvement over
MMLU. This confirms that iteratively updating the expert                                                                                     the baseline. Finally, increasing the evolution rounds from
foundations allows the model to escape local optima and                                                                                      1 to 5 consistently enhances performance. This confirms
integrate knowledge more deeply. Furthermore, the variant                                                                                    that the iterative multi-round process is essential for refining
without cycle loss (w/o Cycle Loss) suffers a significant per-                                                                               expert foundations and achieving higher generalization.
formance drop, particularly in reasoning tasks like NLGraph.
This suggests that the cycle-consistent constraint is vital for
                                                                                                                                             Impact of Model Scale We investigate the scalability of
accurately capturing the high-performance regions of the
                                                                                                                                             EvoGM by varying the number of merged experts from 2
parameter space. Finally, the dual-generator architecture
                                                                                                                                             to 10 in a multi-task setting. Results are summarized in
consistently outperforms the single-generator variant, prov-
                                                                                                                                             Figure 5 and Table 13 in Appendix, where we compare the
ing that bidirectional learning of winner-loser pairs more
                                                                                                                                             initial performance against the final optimized results.
effectively identifies optimal merging coefficients.
                                                                                                                                             Our framework consistently improves average accuracy
Hyperparameter Sensitivity We evaluate the sensitivity                                                                                       across all model counts. When merging 8 models, the av-
of EvoGM to key hyperparameters by merging 3 models                                                                                          erage test score increases from 0.433 to 0.460. This trend

                                                                                                                                         7

## Page 8

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

                                          Average Performance Trend
                                                                                      ter interference. To address this, subsequent studies propose
                                                                                      conflict-aware mechanisms, such as sign-based trimming
                           0.50                                                       (TIES) (Yadav et al., 2023), importance-aware pruning and
                                                                                      rescaling (DU et al., 2024), and the selective removal of re-



      Accuracy (Average)
                                                                                      dundant or conflicting components (Sun et al., 2025a). More
                           0.48
                                                                                      recently, these techniques have been extended to Parameter-
                                                                                      Efficient Fine-Tuning (PEFT) and multimodal settings (Li
                           0.46                                                       et al., 2025c), including LoRA-style merging (Panariello
                                                                                      et al., 2025) and dynamic routing frameworks for heteroge-
                                                      Initial Val
                           0.44                       Final Val
                                                                                      neous architectures (Chen et al., 2025; Lu et al., 2024b).
                                                      Initial Test
                                                      Final Test                      Beyond direct parameter manipulation, another research
                                  2   3     4     5     6       7    8   9   10       thrust focuses on improving model mergeability through
                                                Number of Models
                                                                                      representation-level synchronization. This includes re-
Figure 5. Impact of the number of merged models on performance.                       solving neuron permutation symmetries (Crisostomi et al.,
We evaluate how the merging quality scales when integrating dif-                      2024), enforcing dual-space consistency (Xu et al., 2024b),
ferent quantities of models.
                                                                                      and performing subspace alignment or activation-level cor-
                                                                                      rections (Horoi et al., 2024; Shao et al., 2026; Sun et al.,
                                                                                      2025c; Yao et al., 2025). To further enhance robustness
indicates that EvoGM is highly robust to the initial config-
                                                                                      across heterogeneous task distributions, iterative merging
uration and the complexity of the weight space. Even as
                                                                                      strategies have been proposed to refine weights through
the number of experts increases, the generative evolutionary
                                                                                      multi-stage compositions (Tang et al., 2025; Yuan et al.,
process successfully identifies superior merging coefficients
                                                                                      2025).
that mitigate task interference. These results demonstrate
that our method scales effectively, maintaining consistent                            While early methods rely on hand-crafted heuristics, re-
performance gains regardless of whether a small ensemble                              cent empirical evidence suggests that performance is often
or a larger set of 10 experts is used. This stability proves                          governed by the precision of merging coefficients rather
that EvoGM effectively navigates the high-dimensional pa-                             than the specific operators used (Lan et al., 2025). This
rameter space across various scales of model composition.                             shift in perspective has motivated the use of formal opti-
                                                                                      mization to determine optimal blending weights. Current
Extended Scaling and Robustness Experiments We fur-                                   approaches employ entropy-based objectives (Yang et al.,
ther examine EvoGM under larger model and expert-space                                2024), Bayesian inference (Li et al., 2025b), and Pareto-
settings, with full results reported in Appendix B. To test                           aware multi-objective designs (Chen & Kwok, 2025; Li
model-scale transfer, we merge ten Tulu-finetuned experts                             et al., 2024; Zhou et al., 2025) to explore the weight space.
built on Qwen3-8B; EvoGM achieves the best average score                              However, these methods often rely on stochastic search or
among all compared methods (0.603) while maintaining                                  static probabilistic priors, leaving a gap for more adaptive,
competitive wall-clock search cost. To test scalability with                          generative search mechanisms that can effectively capture
respect to the number of experts, we additionally evaluate a                          high-performance parameter configurations.
20-expert ViT-B-16 merging setting under the same valida-
tion/test split and search budget as CMA-ES, where EvoGM
obtains a higher average test score (0.6363 vs. 0.6268). Fi-                          Evolutionary Model Merging A burgeoning line of re-
nally, multi-seed experiments on Qwen2.5-1.5B show statis-                            search reframes model merging as a gradient-free opti-
tically significant average improvements over PSO-Merging                             mization challenge, moving away from manual heuristic
(p = 0.0100) and Model Swarm (p = 0.0006), indicating                                 design. Evolutionary Algorithms (EAs) have become a
that the performance gains are not due to random variation.                           preferred paradigm due to their flexibility in handling non-
                                                                                      differentiable objectives. One category focuses on opti-
5. Related Work                                                                       mizing merging hyperparameters, such as layer-wise co-
                                                                                      efficients and weight sparsity ratios (Akiba et al., 2025;
Model Merging Model merging facilitates the integration                               Baba et al., 2024). Alternatively, population-centric meth-
of diverse expert models into a unified framework without                             ods evolve the models themselves, employing mutation and
additional training costs (Yang et al., 2026). Early research                         crossover operators to generate novel architectural candi-
demonstrates that models fine-tuned from the same initial-                            dates (Du et al., 2024). More recent swarm-intelligence
ization can be linearly combined when they reside in compat-                          frameworks harness collaborative search trajectories to
ible loss basins (Ilharco et al., 2023; Wortsman et al., 2022).                       identify optimal weight configurations in the parameter
However, direct weight averaging often suffers from parame-                           space (Feng et al., 2025; Zhang et al., 2025).

                                                                                  8

## Page 9

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

Despite their potential, evolutionary model merging meth-            dong Basic and Applied Basic Research Foundation (No.
ods are often bottlenecked by the prohibitive computational          2024B1515020019), in part by National Natural Science
cost of evaluating large-scale models. Scalable evolutionary-        Foundation of China (Grant No. 62502246).
computation platforms such as EvoX (Huang et al., 2024)
have improved the practicality and programmability of evo-           Impact Statement
lutionary search, and efficiency-oriented merging frame-
works further mitigate evaluation costs through surrogate            This work studies training-free model merging for compos-
models or sparse validation subsets (Akizuki et al., 2025;           ing multiple fine-tuned models. By learning to propose
Mencattini et al., 2025). Yet these advances mainly concern          high-performing merge coefficients from historical search
the execution or evaluation of search, leaving the generation        trajectories, EvoGM can reduce the need to retrain full mod-
of candidate merging configurations largely governed by              els or maintain many separate expert models. This makes
generic stochastic operators. These operators often lack the         model adaptation more practical when compute, validation
structural awareness needed to navigate high-dimensional             data, or engineering resources are limited. More broadly,
merging landscapes, leading to poor sample efficiency under          efficient model merging can encourage the reuse of existing
limited evaluation budgets.                                          task-specialized checkpoints, improve the accessibility of
                                                                     customized AI systems, and support more sustainable model
6. Conclusion and Limitations                                        development by amortizing prior fine-tuning costs across
                                                                     multiple downstream tasks.
Conclusion. This paper presents EvoGM, a framework that
                                                                     The societal effects of EvoGM are therefore expected to be
reformulates model merging as a learnable generative task.
                                                                     largely aligned with those of efficient model reuse and com-
By integrating dual-generator learning with a hierarchical
                                                                     position. As with other model-merging methods, a merged
basis shift mechanism, EvoGM effectively navigates the
                                                                     model may inherit biases, privacy concerns, or unsafe be-
complex performance landscape of large language mod-
                                                                     haviors from its source models, so appropriate source-model
els without relying on manual heuristics. Our experiments
                                                                     selection and task-relevant evaluation remain important.
demonstrate that by learning from historical search trajecto-
ries, EvoGM achieves superior data efficiency and discovers
high-synergy model configurations that outperform current            References
state-of-the-art baselines across various benchmarks.                Akiba, T., Shing, M., Tang, Y., Sun, Q., and Ha,
Limitations. The current study focuses on homogeneous                  D.    Evolutionary optimization of model merging
task-vector merging, where all expert models are obtained              recipes.   Nature Machine Intelligence, 7(2):195–
from the same pretrained base model. This setting makes                204, feb 2025. ISSN 2522-5839. doi: 10.1038/
the linear coefficient space well-defined, but it also limits          s42256-024-00975-8. URL https://doi.org/10.
the direct use of EvoGM in more general merging scenar-               1038/s42256-024-00975-8.
ios. For models with different architectures, initializations,
or parameter structures, additional alignment is needed be-          Akizuki, R., Kudo, Y., Yoshinari, N., Hirose, Y., Nishi-
fore coefficient-based search can be applied. Moreover, our            moto, T., Uchida, K., and Shirakawa, S. Surrogate
multi-task setting optimizes a single scalar validation ob-            benchmarks for model merging optimization. In AutoML
jective and returns one deployable merged model. This is              2025 Non-Archival Content Track, 2025. URL https:
suitable when a fixed task balance is given, but it does not          //openreview.net/forum?id=Yv3tRT8olz.
directly provide a set of Pareto-optimal models for users            Baba, K., Yagi, R., Takahashi, J., Kishikawa, R., and Kodera,
with different preferences. Extending EvoGM to heteroge-               S. JRadiEvo: A Japanese radiology report generation
neous model merging and preference-aware multi-objective               model enhanced by evolutionary optimization of model
merging is therefore an important direction for future work.           merging. In Advancements In Medical Foundation Mod-
                                                                       els: Explainability, Robustness, Security, and Beyond,
Acknowledgements                                                       2024. URL https://openreview.net/forum?
                                                                       id=H1osvc7tMP.
This work is supported in part by National Natural Sci-
ence Foundation of China (Grant No. 62276121), in                    Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan,
part by the TianYuan funds for Mathematics of the Na-                  Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical
tional Science Foundation of China (Grant No. 12326604),               report. arXiv preprint arXiv:2309.16609, 2023.
in part by Innovation Team and Talents Cultivation Pro-
gram of National Administration of Traditional Chinese               Bar-Haim, R., Dagan, I., Dolan, B., Ferro, L., Giampiccolo,
Medicine (No: ZYYCXTD-D-202403), in part by Guang-                     D., Magnini, B., and Szpektor, I. The second PASCAL
                                                                       recognising textual entailment challenge. In Proceedings

                                                                 9

## Page 10

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

  of the Second PASCAL Challenges Workshop on Recog-                   F. (eds.), Machine Learning Challenges. Evaluating Pre-
  nising Textual Entailment, 2006. URL https://api.                    dictive Uncertainty, Visual Object Classification, and
  semanticscholar.org/CorpusID:13385138.                               Recognising Tectual Entailment, pp. 177–190, Berlin,
                                                                       Heidelberg, 2006. Springer Berlin Heidelberg. ISBN
Bentivogli, L., Clark, P., Dagan, I., and Giampiccolo, D.              978-3-540-33428-6.
  The fifth pascal recognizing textual entailment challenge.
  TAC, 7(8):1, 2009.                                                 Deep, P. T., Bhardwaj, R., and Poria, S. DELLA-
                                                                       Merging: Reducing interference in model merging
Cer, D., Diab, M., Agirre, E., Lopez-Gazpio, I., and                   through magnitude-based sampling. arXiv preprint
  Specia, L. SemEval-2017 task 1: Semantic textual                     arXiv:2406.11617, abs/2406.11617, 2024. URL https:
  similarity multilingual and crosslingual focused evalu-             //doi.org/10.48550/arXiv.2406.11617.
  ation. In Bethard, S., Carpuat, M., Apidianaki, M.,
  Mohammad, S. M., Cer, D., and Jurgens, D. (eds.),                  Ding, W., Feng, S., Liu, Y., Tan, Z., Balachandran, V.,
  Proceedings of the 11th International Workshop on Se-                He, T., and Tsvetkov, Y. Knowledge crosswords:
  mantic Evaluation (SemEval-2017), pp. 1–14, Vancou-                  Geometric knowledge reasoning with large language
  ver, Canada, August 2017. Association for Computa-                   models. In Findings of the Association for Com-
  tional Linguistics. doi: 10.18653/v1/S17-2001. URL                   putational Linguistics: ACL 2024, pp. 2609–2636,
  https://aclanthology.org/S17-2001/.                                  2024. URL https://doi.org/10.18653/v1/
                                                                       2024.findings-acl.154.
Chaudhary, S. Code Alpaca: An instruction-following
  LLaMA model for code generation.      https://                     Dolan, W. B. and Brockett, C. Automatically construct-
  github.com/sahil280114/codealpaca, 2023.                             ing a corpus of sentential paraphrases. In Proceedings
                                                                       of the Third International Workshop on Paraphrasing
Chen, S., Zhang, J., Zhu, T., Liu, W., Gao, S., Xiong, M.,            (IWP2005), 2005. URL https://aclanthology.
  Li, M., and He, J. Bring reason to vision: Understand-               org/I05-5002/.
  ing perception and reasoning through model merging.
  In Forty-second International Conference on Machine                DU, G., Lee, J., Li, J., Jiang, R., Guo, Y., Yu, S., Liu,
  Learning, 2025. URL https://openreview.net/                         H., Goh, S. K., Tang, H.-K., He, D., and Zhang, M.
  forum?id=ntCAP6tMoX.                                                Parameter competition balancing for model merging.
                                                                      In The Thirty-eighth Annual Conference on Neural In-
Chen, W. and Kwok, J. Pareto Merging: Multi-objective op-             formation Processing Systems, 2024. URL https:
  timization for preference-aware model merging. In Forty-            //openreview.net/forum?id=l5SbrtvSRS.
  second International Conference on Machine Learning,
  2025. URL https://openreview.net/forum?                            Du, G., Li, J., Liu, H., Jiang, R., Yu, S., Guo, Y., Goh,
  id=D7qRwx6BOS.                                                       S. K., and Tang, H.-K. Knowledge fusion by evolving
                                                                      weights of language models. In Findings of the Associa-
Chung, H. W., Hou, L., Longpre, S., Zoph, B., Tay, Y., Fe-             tion for Computational Linguistics: ACL 2024, volume
  dus, W., Li, Y., Wang, X., Dehghani, M., Brahma, S., et al.          abs/2406.12208, 2024. URL https://doi.org/10.
  Scaling instruction-finetuned language models. Journal              48550/arXiv.2406.12208.
  of Machine Learning Research, 25(70):1–53, 2024.
                                                                     Feng, S., Wang, Z., Wang, Y., Ebrahimi, S., Palangi, H.,
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H.,              Miculicich, L., Kulshrestha, A., Rauschmayr, N., Choi,
  Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano,            Y., Tsvetkov, Y., et al. Model swarms: Collaborative
  R., Hesse, C., and Schulman, J. Training verifiers to solve          search to adapt llm experts via swarm intelligence. In In-
  math word problems. arXiv preprint arXiv:2110.14168,                 ternational Conference on Machine Learning, pp. 16904–
  abs/2110.14168, 2021. URL https://arxiv.org/                         16930. PMLR, 2025.
  abs/2110.14168.
                                                                     Gao, H., Kaltenbach, S., and Koumoutsakos, P. Generative
Crisostomi, D., Fumero, M., Baieri, D., Bernard, F., and               learning for forecasting the dynamics of high-dimensional
  Rodolà, E. C2 M3 : Cycle-consistent multi-model merg-               complex systems. Nature Communications, 15(1):8904,
  ing. In The Thirty-eighth Annual Conference on Neural                2024. doi: 10.1038/s41467-024-53165-w.
  Information Processing Systems, 2024. URL https:
  //openreview.net/forum?id=iD18l6prA7.                              Gehman, S., Gururangan, S., Sap, M., Choi, Y., and Smith,
                                                                       N. A. RealToxicityPrompts: Evaluating neural toxic
Dagan, I., Glickman, O., and Magnini, B. The PASCAL                    degeneration in language models. In Cohn, T., He, Y.,
  recognising textual entailment challenge. In Quiñonero-             and Liu, Y. (eds.), Findings of the Association for Com-
  Candela, J., Dagan, I., Magnini, B., and d’Alché Buc,               putational Linguistics: EMNLP 2020, pp. 3356–3369,

                                                                10

## Page 11

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

  Online, November 2020. Association for Computational                 Jiang, H., Wang, R., Liang, W., Sun, Q., Zhang, X., and
  Linguistics. doi: 10.18653/v1/2020.findings-emnlp.                      Liu, Y. Slerp-Opt: merging large language models via
  301. URL https://aclanthology.org/2020.                                 adaptive strategies. The Journal of Supercomputing, 81
  findings-emnlp.301/.                                                    (12):1223, 08 2025. ISSN 1573-0484. doi: 10.1007/
                                                                          s11227-025-07727-4. URL https://doi.org/10.
Giampiccolo, D., Magnini, B., Dagan, I., and Dolan, B.                    1007/s11227-025-07727-4.
  The third PASCAL recognizing textual entailment chal-
  lenge. In Sekine, S., Inui, K., Dagan, I., Dolan, B.,                Jiang, T., Sun, K., Liang, Z., Cheng, R., Jin, Y., and Tan,
  Giampiccolo, D., and Magnini, B. (eds.), Proceedings                    K. C. Evolutionary generative optimization: Towards
  of the ACL-PASCAL Workshop on Textual Entailment                        fully data-driven evolutionary optimization via generative
  and Paraphrasing, pp. 1–9, Prague, June 2007. Asso-                     learning. IEEE Transactions on Evolutionary Computa-
  ciation for Computational Linguistics. URL https:                       tion, 2026.
  //aclanthology.org/W07-1401/.                                        Jin, R., Hou, B., Xiao, J., Su, W. J., and Shen, L. Fine-
                                                                          tuning attention modules only: Enhancing weight dis-
Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M.,
                                                                          entanglement in task arithmetic. In The Thirteenth In-
  Song, D., and Steinhardt, J. Measuring massive multitask
                                                                          ternational Conference on Learning Representations,
  language understanding. In International Conference
                                                                          2025. URL https://openreview.net/forum?
  on Learning Representations, 2021. URL https://
                                                                          id=dj0TktJcVI.
  openreview.net/forum?id=d7KBjmI3GmQ.
                                                                       Jin, X., Ren, X., Preotiuc-Pietro, D., and Cheng, P. Data-
Horoi, S., Camacho, A. M. O., Belilovsky, E., and Wolf,                   less knowledge fusion by merging weights of language
  G. Harmony in diversity: Merging neural networks with                   models. In The Eleventh International Conference
  canonical correlation analysis. In Forty-first International            on Learning Representations, 2023. URL https://
 Conference on Machine Learning, 2024. URL https:                         openreview.net/forum?id=FCnohuR6AnM.
 //openreview.net/forum?id=hLuNVjRnY3.
                                                                       Köpf, A., Kilcher, Y., Von Rütte, D., Anagnostidis, S.,
Hu, E. J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y.,               Tam, Z. R., Stevens, K., Barhoum, A., Nguyen, D., Stan-
 Wang, S., Wang, L., and Chen, W. LoRA: Low-rank adap-                   ley, O., Nagyfi, R., et al. Openassistant conversations-
  tation of large language models. In International Confer-              democratizing large language model alignment. Ad-
  ence on Learning Representations, 2022. URL https:                     vances in neural information processing systems, 36:
 //openreview.net/forum?id=nZeVKeeFYf9.                                  47669–47681, 2023.

Huang, B., Cheng, R., Li, Z., Jin, Y., and Tan, K. C. Evox: A          Lan, X., Zheng, Y., Cao, S., and Li, Y. The think-
  distributed gpu-accelerated framework for scalable evolu-              ing spectrum: An empirical study of tunable rea-
  tionary computation. IEEE Transactions on Evolutionary                 soning in LLMs through model merging.        arXiv
  Computation, 2024.                                                     preprint arXiv:2509.22034, abs/2509.22034, Septem-
                                                                         ber 2025. URL https://doi.org/10.48550/
Ilharco, G., Ribeiro, M. T., Wortsman, M., Schmidt, L.,                  arXiv.2509.22034.
   Hajishirzi, H., and Farhadi, A. Editing models with task
                                                                       Levesque, H. J., Davis, E., and Morgenstern, L. The wino-
   arithmetic. In The Eleventh International Conference
                                                                         grad schema challenge. In AAAI Spring Symposium:
   on Learning Representations, 2023. URL https://
                                                                         Logical Formalizations of Commonsense Reasoning,
   openreview.net/forum?id=6t0Kwf8-jrj.
                                                                         2011. URL https://api.semanticscholar.
Inoshita, K., Zhou, X., and Kawai, A. Multi-stage evo-                   org/CorpusID:15710851.
  lutionary model merging with meta data driven cur-                   Li, B., Di, Z., Yang, Y., Qian, H., Yang, P., Hao, H., Tang,
  riculum learning for sentiment-specialized large lan-                  K., and Zhou, A. It’s morphing time: Unleashing the po-
  guage modeling. 2024 IEEE International Confer-                        tential of multiple LLMs via multi-objective optimization.
  ence on Data Science and Systems (DSS), pp. 58–65,                     IEEE Transactions on Evolutionary Computation, 2025a.
  2024. URL https://api.semanticscholar.                                 doi: 10.1109/TEVC.2025.3613937.
  org/CorpusID:284648786.
                                                                       Li, L., Zhang, T., Bu, Z., Wang, S., He, H., Fu, J., Wu, Y.,
Ivison, H., Wang, Y., Pyatkin, V., Lambert, N., Peters, M.,              Bian, J., Chen, Y., and Bengio, Y. MAP: Model merging
   Dasigi, P., Jang, J., Wadden, D., Smith, N. A., Beltagy, I.,          with amortized pareto front using limited computation. In
   et al. Camels in a changing climate: Enhancing LM adap-               International Workshop on Federated Foundation Models
   tation with TULU 2. arXiv preprint arXiv:2311.10702,                  in Conjunction with NeurIPS 2024, 2024. URL https:
   2023.                                                                 //openreview.net/forum?id=KfOdVp4pfm.

                                                                  11

## Page 12

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

Li, M., Lu, Y., Dai, Q., Huang, S., Ding, Y., and Lu,                 August 2024. Association for Computational Linguis-
  H. BECAME: Bayesian continual learning with adap-                   tics. doi: 10.18653/v1/2024.acl-long.445. URL https:
  tive model merging. In Forty-second International Con-              //aclanthology.org/2024.acl-long.445/.
  ference on Machine Learning, 2025b. URL https:
  //openreview.net/forum?id=gU0MwTihsn.                             Ma, Q., Liu, D., Chen, Q., Zhang, L., and Shao, J. LED-
                                                                     Merging: Mitigating safety-utility conflicts in model
Li, X. L. and Liang, P.      Prefix-Tuning: Optimiz-                 merging with location-election-disjoint. In Proceedings
  ing continuous prompts for generation.    Proceed-                 of the 63rd Annual Meeting of the Association for Compu-
  ings of the 59th Annual Meeting of the Association                 tational Linguistics (Volume 1: Long Papers), pp. 21749–
  for Computational Linguistics and the 11th Interna-                21767, 2025. URL https://aclanthology.org/
  tional Joint Conference on Natural Language Pro-                   2025.acl-long.1055/.
  cessing (Volume 1: Long Papers), pp. 4582–4597,
  2021. URL https://api.semanticscholar.                            Matena, M. S. and Raffel, C. A. Merging models with
  org/CorpusID:230433941.                                            Fisher-weighted averaging. Advances in Neural Informa-
                                                                     tion Processing Systems, 35:17703–17716, 2022.
Li, Y., Lan, X., Chen, H., Lu, K., and Jiang, D. Multi-
  modal pear chain-of-thought reasoning for multimodal              Mencattini, T., Minut, A. R., Crisostomi, D., Santilli, A.,
  sentiment analysis. ACM Transactions on Multimedia                 and Rodolà, E. Merge3 : Efficient evolutionary merg-
  Computing, Communications and Applications, 20(9):                 ing on consumer-grade gpus. In Proceedings of the
  1–23, 2025c.                                                       42nd International Conference on Machine Learning
                                                                     (ICML), 2025. URL https://openreview.net/
Lian, W., Goodson, B., Pentland, E., Cook, A.,                       pdf?id=qFXDv0X4yc.
  Vong, C., and ”Teknium”.   Openorca: An open
  dataset of gpt augmented flan reasoning traces.                   Panariello, A., Marczak, D., Magistri, S., Porrello, A.,
  https://https://huggingface.co/                                     Twardowski, B., Bagdanov, A. D., Calderara, S., and
  datasets/Open-Orca/OpenOrca, 2023.                                  van de Weijer, J. Accurate and efficient low-rank model
                                                                      merging in core space. In The Thirty-ninth Annual
Lin, S., Hilton, J., and Evans, O. Truthfulqa: Measuring how          Conference on Neural Information Processing Systems,
  models mimic human falsehoods, 2022. URL https://arxiv.             2025. URL https://openreview.net/forum?
  org/abs/2109.07958, 1, 2021.                                        id=y1z7SAS8q8.

Liu, D., Wang, Z., Wang, B., Chen, W., Li, C., Tu, Z., Chu,         Perin, G., Chen, X., Liu, S., Kailkhura, B., Wang, Z., and
  D., and Sui, D. Maximizing intermediate checkpoint                  Gallagher, B. RankMean: Module-level importance score
  value in LLM pretraining with bayesian optimization.                for merging fine-tuned LLM models. In Ku, L.-W., Mar-
  In Forty-second International Conference on Machine                 tins, A., and Srikumar, V. (eds.), Findings of the Associa-
  Learning, 2025. URL https://openreview.net/                         tion for Computational Linguistics: ACL 2024, pp. 1776–
  forum?id=UvwWrUV1JV.                                                1782, Bangkok, Thailand, August 2024. Association
                                                                      for Computational Linguistics. doi: 10.18653/v1/2024.
Lu, J., Pang, Z., Xiao, M., Zhu, Y., Xia, R., and Zhang,              findings-acl.104. URL https://aclanthology.
  J. Merge, ensemble, and cooperate! a survey on col-                 org/2024.findings-acl.104/.
  laborative strategies in the era of large language mod-
  els. arXiv preprint arXiv:2407.06089, 2024a. URL                  Shao, Z., Xiong, B., Yang, X., Song, Y., Zhang, Q., Chen,
  https://arxiv.org/abs/2407.06089.                                   H., and Xu, C. Pivotmerge: Bridging heterogeneous mul-
                                                                      timodal pre-training via post-alignment model merging.
Lu, Z., Fan, C., Wei, W., Qu, X., Chen, D., and Cheng,                arXiv preprint arXiv:2604.22823, 2026.
  Y. Twin-Merging: Dynamic integration of modular ex-
  pertise in model merging. In The Thirty-eighth Annual             Socher, R., Perelygin, A., Wu, J., Chuang, J., Manning,
  Conference on Neural Information Processing Systems,                C. D., Ng, A. Y., and Potts, C. Recursive deep models for
  2024b. URL https://openreview.net/forum?                            semantic compositionality over a sentiment treebank. In
  id=81YIt63TTn.                                                      Proceedings of the 2013 conference on empirical methods
                                                                      in natural language processing, pp. 1631–1642, 2013.
Lv, K., Yang, Y., Liu, T., Guo, Q., and Qiu, X. Full pa-
  rameter fine-tuning for large language models with lim-           Sun, W., Li, Q., Geng, Y., and Li, B. CAT merg-
  ited resources. In Ku, L.-W., Martins, A., and Srikumar,            ing: A training-free approach for resolving conflicts
  V. (eds.), Proceedings of the 62nd Annual Meeting of                in model merging. In Forty-second International Con-
  the Association for Computational Linguistics (Volume               ference on Machine Learning, 2025a. URL https:
  1: Long Papers), pp. 8187–8198, Bangkok, Thailand,                  //openreview.net/forum?id=zy7Jw91tdh.

                                                               12

## Page 13

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

Sun, W., Li, Q., Wang, W., Geng, Y., and Li, B. Task                  Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S.,
  arithmetic in trust region: A training-free model merg-              Ren, W., Arulraj, A., He, X., Jiang, Z., et al. MMLU-Pro:
  ing approach to navigate knowledge conflicts. In                     A more robust and challenging multi-task language un-
  Proceedings of the 33rd ACM International Confer-                    derstanding benchmark. Advances in Neural Information
  ence on Multimedia, MM ’25, pp. 5178–5187, New                       Processing Systems, 37:95266–95290, 2024.
  York, NY, USA, 2025b. Association for Computing
  Machinery. ISBN 9798400720352. doi: 10.1145/                        Warstadt, A., Singh, A., and Bowman, S. R. Neural network
  3746027.3755789. URL https://doi.org/10.                             acceptability judgments. Transactions of the Association
  1145/3746027.3755789.                                                for Computational Linguistics, 7:625–641, 2019.

Sun, W., Li, Q., Wang, W., Liu, Y., Geng, Y., and Li, B. To-          Wei, J., Wang, X., Schuurmans, D., Bosma, M., Ichter, B.,
  wards minimizing feature drift in model merging: Layer-              Xia, F., Chi, E. H., Le, Q. V., and Zhou, D. Chain-of-
  wise task vector fusion for adaptive knowledge integra-              thought prompting elicits reasoning in large language
  tion. In The Thirty-ninth Annual Conference on Neural                models. In Proceedings of the 36th International Confer-
  Information Processing Systems, 2025c. URL https:                    ence on Neural Information Processing Systems, NIPS
  //openreview.net/forum?id=0KOfAUiHua.                                ’22, Red Hook, NY, USA, 2022. Curran Associates Inc.
                                                                       ISBN 9781713871088.
Tang, A., Shen, L., Luo, Y., Zhan, Y., Hu, H., Du, B.,
                                                                      Williams, A., Nangia, N., and Bowman, S. A broad-
  Chen, Y., and Tao, D. Parameter-efficient multi-task
                                                                       coverage challenge corpus for sentence understanding
  model fusion with partial linearization. The Twelfth
                                                                       through inference. In Walker, M., Ji, H., and Stent,
  International Conference on Learning Representations,
                                                                       A. (eds.), Proceedings of the 2018 Conference of the
  abs/2310.04742, 2024. URL https://doi.org/10.
                                                                       North American Chapter of the Association for Com-
  48550/arXiv.2310.04742.
                                                                       putational Linguistics: Human Language Technologies,
                                                                       Volume 1 (Long Papers), pp. 1112–1122, New Orleans,
Tang, A., Yang, E., Shen, L., Luo, Y., Hu, H., Zhang, L.,
                                                                       Louisiana, June 2018. Association for Computational
  Du, B., and Tao, D. Merging on the fly without retraining:
                                                                       Linguistics. doi: 10.18653/v1/N18-1101. URL https:
  A sequential approach to scalable continual model merg-
                                                                       //aclanthology.org/N18-1101/.
  ing. In The Thirty-ninth Annual Conference on Neural
  Information Processing Systems, 2025. URL https:                    Wortsman, M., Ilharco, G., Gadre, S. Y., Roelofs, R., Lopes,
  //openreview.net/forum?id=rdGMyTPhui.                                R. G., Morcos, A. S., Namkoong, H., Farhadi, A., Car-
                                                                       mon, Y., Kornblith, S., and Schmidt, L. Model soups: av-
Wang, A., Singh, A., Michael, J., Hill, F., Levy, O., and
                                                                       eraging weights of multiple fine-tuned models improves
 Bowman, S. GLUE: A multi-task benchmark and analy-
                                                                       accuracy without increasing inference time. In Interna-
 sis platform for natural language understanding. In Pro-
                                                                       tional conference on machine learning, pp. 23965–23998,
 ceedings of the 2018 EMNLP workshop BlackboxNLP:
                                                                       2022. URL https://proceedings.mlr.press/
 Analyzing and interpreting neural networks for NLP, pp.
                                                                       v162/wortsman22a.html.
 353–355, 2018.
                                                                      Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J.,
Wang, F., Xie, J., Zhou, A., and Tang, K. A new prediction             Tao, C., Lin, Q., and Jiang, D. WizardLM: Empower-
 strategy for dynamic multiobjective optimization using                 ing large pre-trained language models to follow com-
 diffusion model. IEEE Transactions on Evolutionary                     plex instructions. In The Twelfth International Confer-
 Computation, 29(5):1575–1589, 2025. doi: 10.1109/                      ence on Learning Representations, 2024a. URL https:
 TEVC.2025.3551323.                                                    //openreview.net/forum?id=CfXh93NDgH.
Wang, H., Feng, S., He, T., Tan, Z., Han, X., and Tsvetkov,           Xu, Z., Yuan, K., Wang, H., Wang, Y., Song, M., and Song, J.
 Y. Can language models solve graph problems in natural                 Training-free pretrained model merging. In Proceedings
 language? Advances in Neural Information Processing                    of the IEEE/CVF Conference on Computer Vision and
 Systems, 36:30840–30861, 2023.                                         Pattern Recognition (CVPR), pp. 5915–5925, June 2024b.

Wang, J., Liu, Q., Liang, H., Joshi, G., and Poor, H. V. Tack-        Yadav, P., Tam, D., Choshen, L., Raffel, C. A., and Bansal,
 ling the objective inconsistency problem in heterogeneous              M. TIES-merging: Resolving interference when merging
 federated optimization. In Proceedings of the 34th Inter-              models. In Oh, A., Naumann, T., Globerson, A., Saenko,
 national Conference on Neural Information Processing                   K., Hardt, M., and Levine, S. (eds.), Advances in Neural
 Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran                     Information Processing Systems, volume 36, pp. 7093–
 Associates Inc. ISBN 9781713829546.                                    7115. Curran Associates, Inc., 2023.

                                                                 13

## Page 14

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

Yang, E., Wang, Z., Shen, L., Liu, S., Guo, G., Wang, X.,             of the Conference on Empirical Methods in Natural
  and Tao, D. AdaMerging: Adaptive model merging for                  Language Processing (EMNLP 2024), pp. 2289–2305,
  multi-task learning. In The Twelfth International Confer-           2024. URL https://aclanthology.org/2024.
  ence on Learning Representations, 2024. URL https:                  findings-emnlp.127.
  //openreview.net/forum?id=nZP6NgD3QY.
                                                                    Zheng, H., Shen, L., Tang, A., Luo, Y., Hu, H., Du,
Yang, E., Shen, L., Guo, G., Wang, X., Cao, X., Zhang,                B., Wen, Y., and Tao, D. Learning from models be-
  J., and Tao, D. Model merging in LLMs, MLLMs,                       yond fine-tuning. Nature Machine Intelligence, 7(1):
  and beyond: Methods, theories, applications, and op-                6–17, 01 2025. ISSN 2522-5839. doi: 10.1038/
  portunities. ACM Computing Surveys, January 2026.                   s42256-024-00961-0. URL https://doi.org/10.
  ISSN 0360-0300. doi: 10.1145/3787849. URL https:                    1038/s42256-024-00961-0.
  //doi.org/10.1145/3787849. Just Accepted.
                                                                    Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X.,
Yang, Z., Qi, B., Sun, H., Long, W., Zhao, R., and                    Efrat, A., Yu, P., Yu, L., Zhang, S., Ghosh, G., Lewis,
  Gao, X. CABS: Conflict-aware and balanced sparsi-                   M., Zettlemoyer, L., and Levy, O. LIMA: less is more
  fication for enhancing model merging. arXiv preprint                for alignment. In Proceedings of the 37th International
  arXiv:2503.01874, 2025.                                             Conference on Neural Information Processing Systems,
                                                                      NIPS ’23, Red Hook, NY, USA, 2023. Curran Associates
Yao, Y., LIU, S., Liu, Z., Li, Q., LIU, M., Han, X., Guo,             Inc.
  Z., Wu, H., and Song, L. Activation-guided consensus
  merging for large language models. In The Thirty-ninth            Zhou, Y., Wu, X., Wu, J., Feng, L., and Tan, K. HM3: Hi-
  Annual Conference on Neural Information Processing                  erarchical multi-objective model merging for pretrained
  Systems, 2025. URL https://openreview.net/                          models. In The Thirty-ninth Annual Conference on Neu-
  forum?id=ayzWTxb9ZD.                                                ral Information Processing Systems, 2025. URL https:
                                                                      //openreview.net/forum?id=JeP0lpusYw.
Yu, L., Yu, B., Yu, H., Huang, F., and Li, Y. Extend
  model merging from fine-tuned to pre-trained large lan-
  guage models via weight disentanglement. arXiv preprint
  arXiv:2408.03092, 2024a. URL https://arxiv.
  org/abs/2408.03092.

Yu, L., Yu, B., Yu, H., Huang, F., and Li, Y. Language
  models are super mario: Absorbing abilities from homol-
  ogous models as a free lunch. In Forty-first International
  Conference on Machine Learning, 2024b.

Yuan, X., Zhang, C., Liu, Z., Shi, D., Vosoughi, S., and
  Lee, W. Superficial self-improved reasoners benefit from
  model merging. Proceedings of the 2025 Conference
  on Empirical Methods in Natural Language Processing,
  abs/2503.02103:5912–5932, March 2025. URL https:
  //doi.org/10.48550/arXiv.2503.02103.

Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y.
  HellaSwag: Can a machine really finish your sentence?
  In Annual Meeting of the Association for Computational
  Linguistics, pp. 4791–4800, 2019. URL https://doi.
  org/10.18653/v1/p19-1472.

Zhang, K., Zhang, S., and Feng, Y. PSO-Merging: Merging
  models based on particle swarm optimization. arXiv
  preprint arXiv:2508.19839, 2025. URL https://
  arxiv.org/abs/2508.19839.

Zhang, Y., Wang, H., Feng, S., Tan, Z., Han, X., He,
  T., and Tsvetkov, Y. Can LLM graph reasoning gen-
  eralize beyond pattern memorization?   In Findings

                                                               14

## Page 15

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

A. Complexity Analysis
To evaluate the efficiency of EvoGM, we analyze its computational overhead in terms of both time and space complexity.
Let N denote the number of expert models, D the number of parameters per model (or LoRA rank), R the number of outer
rounds, T the inner evolutionary iterations, P the population size, and E the cost of a single evaluation on the validation set.


Time Complexity. The time complexity of EvoGM is primarily dominated by the iterative evaluation and expert refinement
phases. Initially, computing task vectors requires O(N D) operations in the parameter space. During the main loop, the
generative optimization occurs in the low-dimensional coefficient space with a negligible cost of O(RT P · poly(N )),
as N ≪ D. The most significant bottleneck is the fitness evaluation of candidate models, which sequentially takes
O(RT P E). However, since evaluations of candidate coefficients are embarrassingly parallel, the total wall-clock time can
be significantly reduced by employing M parallel workers. In such a parallelized setting, the time complexity is optimized
to O(N D + RT ⌈P/M ⌉E + RKN D), where the final term accounts for the K model merging operations per round during
expert refinement. This linear scalability with respect to M allows EvoGM to handle large populations and intensive search
rounds efficiently.


Space Complexity. In terms of space complexity, EvoGM is designed to be highly memory-efficient, especially within
parameter-efficient fine-tuning (PEFT) frameworks. The primary memory footprint consists of the frozen backbone model
and N task vectors, resulting in a total requirement of O(Dbase + N Dtask ). Throughout the evolutionary process, the
algorithm only needs to store a history of population coefficients and their corresponding scores, which occupies O(RT P N )
space. Given that N is typically small, this storage cost is marginal compared to the model parameters. During the evaluation
phase, EvoGM avoids the need for multiple full-model replicas by dynamically merging task vectors in-place, maintaining a
consistent space complexity of O(D + N D). This efficiency ensures that our approach remains feasible even for large-scale
language models with limited hardware resources.

B. Additional Robustness and Scaling Results
For the experiments reported in this section, we used Ascend 910 NPUs together with HiSilicon AArch64 processors. The
other experiments were conducted on NVIDIA A100-SXM4-40GB GPUs with AMD EPYC 7742 64-Core processors.
Since these platforms differ in their underlying hardware architectures, minor discrepancies in absolute performance may
arise. However, within each experimental setting, our method and all baselines were executed on identical hardware, thereby
ensuring the fairness of the comparative evaluation.

B.1. Statistical Robustness
To assess whether small average gains are driven by random variation, we reran EvoGM, PSO-Merging, and Model Swarm
with matched random seeds on the Qwen2.5-1.5B unseen-task benchmark. Table 4 reports mean and standard deviation over
four runs and independent two-sample Welch’s t-tests. While not every task-level difference is significant, EvoGM obtains a
statistically significant improvement in the average score over PSO-Merging (p = 0.0100) and Model Swarm (p = 0.0006).

                              Table 4. Multi-seed statistical robustness on Qwen2.5-1.5B unseen tasks.

  Task                         EvoGM              PSO-Merging         Model Swarm         EvoGM vs. PSO   EvoGM vs. Swarm
                              mean ± std           mean ± std          mean ± std            p-value          p-value
  MMLU                     0.5942 ± 0.0184       0.5990 ± 0.0072     0.6015 ± 0.0163           0.6614            0.5770
  MMLU-Pro                 0.2412 ± 0.0084       0.2298 ± 0.0072     0.2400 ± 0.0139           0.0840            0.8839
  HellaSwag                0.5930 ± 0.0074       0.5925 ± 0.0115     0.5900 ± 0.0262           0.9446            0.8377
  Knowledge Crosswords     0.4080 ± 0.0041       0.3853 ± 0.0418     0.3948 ± 0.0068           0.3563            0.0211
  GSM8K                    0.4408 ± 0.0256       0.3675 ± 0.0454     0.3295 ± 0.0070           0.0400            0.0021
  NLGraph                  0.5262 ± 0.0184       0.5443 ± 0.0506     0.3748 ± 0.0005           0.5420            0.0005
  TruthfulQA               0.4344 ± 0.0159       0.4315 ± 0.0184     0.4303 ± 0.0148           0.8234            0.7220
  MMLU-Abstain             0.1333 ± 0.0223       0.1137 ± 0.0172     0.1330 ± 0.0265           0.2190            0.9890
  Average                  0.4214 ± 0.0073       0.4018 ± 0.0019     0.3867 ± 0.0076           0.0100            0.0006




                                                                15

## Page 16

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

B.2. Fairness of the Basis Shift
To test whether EvoGM benefits merely from being granted a multi-round basis shift, we also applied the same two-round,
three-iteration-per-round schedule to PSO-Merging. As shown in Table 5, simply adding the same basis-reset mechanism to
PSO-Merging reduces the average score from 0.4018 to 0.3425. This indicates that the basis shift is not a generic advantage
for any search method; in EvoGM, it works together with learned winner-loser mappings and cycle consistency to preserve
useful search structure across rounds.
    Table 5. Single-round PSO, multi-round PSO with the same basis-reset schedule, and multi-round EvoGM on Qwen2.5-1.5B.

                   Task                      PSO Single-Round           PSO Multi-Round     EvoGM Multi-Round
                   MMLU                            0.5990                   0.5660                   0.5942
                   MMLU-Pro                        0.2298                   0.2070                   0.2412
                   HellaSwag                       0.5925                   0.5840                   0.5930
                   Knowledge Crosswords            0.3853                   0.3790                   0.4080
                   GSM8K                           0.3675                   0.2060                   0.4408
                   NLGraph                         0.5443                   0.2820                   0.5262
                   TruthfulQA                      0.4315                   0.4279                   0.4344
                   MMLU-Abstain                    0.1137                   0.0880                   0.1333
                   Average                         0.4018                   0.3425                   0.4214



B.3. Scaling and Search Cost on Qwen3-8B
We further evaluate EvoGM on Qwen3-8B with ten Tulu-finetuned experts. Table 6 shows that EvoGM achieves the best
average score among the tested methods. We also report wall-clock time and forward-pass counts in Table 7. Although
EvoGM uses more candidate evaluations than Model Swarm in this setup, the total wall-clock time remains competitive
because candidate evaluation is parallelizable and the small MLP generators add negligible overhead relative to model
inference.
                                Table 6. Qwen3-8B test performance with ten Tulu-finetuned experts.

 Method             MMLU       MMLU-Pro       HellaSwag     K-Cross        GSM8K        NLGraph     TruthQA   AbstainQA    AVG
 Base                0.655        0.349          0.792          0.589       0.125        0.534        0.561        0.308   0.489
 MTL                 0.702        0.399          0.712          0.667       0.428        0.363        0.558        0.296   0.516
 Single Best         0.732        0.432          0.802          0.644       0.614        0.548        0.630        0.400   0.600
 Task Arithmetic     0.721        0.432          0.809          0.580       0.355        0.413        0.566        0.232   0.513
 TIES Merging        0.673        0.369          0.805          0.558       0.214        0.457        0.571        0.349   0.500
 Model Swarm         0.712        0.420          0.789          0.669       0.588        0.551        0.637        0.381   0.593
 EvoGM               0.718        0.429          0.824          0.586       0.655        0.607        0.614        0.394   0.603


Table 7. Search cost on Qwen3-8B using the Ascend 910 NPUs together with HiSilicon AArch64 processors. FPs denotes candidate
forward-pass evaluations on validation data.

                    Task           EvoGM Time       EvoGM FPs           Model Swarm Time         Model Swarm FPs
                    MMLU              43.1 min            160                32.5 min                  100
                    MMLU-Pro          40.4 min            160                26.2 min                  100
                    HellaSwag         38.4 min            160                31.0 min                  120
                    K-Cross           37.7 min            160                40.0 min                  120
                    GSM8K             86.5 min            160                98.2 min                  100
                    NLGraph           58.5 min            160                68.3 min                  100
                    TruthQA           38.7 min            160                23.0 min                  100
                    AbstainQA         42.2 min            160               109.7 min                  160
                    Total            385.5 min           1280               428.9 min                  900



B.4. Higher-Dimensional Expert Space
To probe a larger coefficient space, we compare EvoGM with CMA-ES on a ViT-B-16 model merging benchmark with
20 experts. Both methods use the same validation/test subsets and the same six-iteration search budget. Table 8 shows

                                                                  16

## Page 17

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

that EvoGM obtains a higher average test score than CMA-ES (0.6363 vs. 0.6268). It suggests that the generative search
mechanism remains useful beyond the 8-10 expert settings used in the main experiments.

                       Table 8. ViT-B-16 merging with 20 experts under an identical six-iteration budget.

                                           Task               CMA-ES       EvoGM
                                           Cars                 0.5028      0.7006
                                           DTD                  0.6528      0.5111
                                           EuroSAT              0.4828      0.8106
                                           GTSRB                0.7794      0.5061
                                           MNIST                0.7050      0.5894
                                           RESISC45             0.6767      0.7367
                                           SVHN                 0.8956      0.4556
                                           SUN397               0.6206      0.6783
                                           STL10                0.9622      0.9783
                                           OxfordIIITPet        0.9189      0.8439
                                           Flowers102           0.6172      0.6217
                                           CIFAR100             0.7300      0.7611
                                           PCAM                 0.5744      0.5706
                                           FER2013              0.4922      0.4394
                                           CIFAR10              0.8511      0.9433
                                           Food101              0.6406      0.8867
                                           FashionMNIST         0.5128      0.8044
                                           RenderedSST2         0.4872      0.6433
                                           EMNIST               0.2939      0.1294
                                           KMNIST               0.1406      0.1161
                                           Average              0.6268      0.6363



C. Experimental Details
C.1. Baselines
We select a total of 13 representative model merging methods as baselines. The details of these methods are summarized
below.

  • Model Soup: This approach merges multiple fine-tuned models by uniformly averaging their parameters (Wortsman
    et al., 2022). It assumes that independently trained models lie in a shared low-loss basin, enabling effective aggregation
    without additional tuning.

  • TA (Task Arithmetic): Task Arithmetic (Ilharco et al., 2023) represents task-specific adaptations as parameter
    difference vectors relative to a base model and combines them through linear addition. This formulation enables
    compositional transfer across tasks but may suffer from interference under naive aggregation.

  • DARE: DARE mitigates parameter conflicts by sparsifying task vectors before aggregation (Yu et al., 2024b). By
    retaining only high-magnitude parameter updates, it reduces destructive interference during merging.

  • TIES: Rather than averaging all parameters, this method resolves sign conflicts across task vectors and selectively trims
    inconsistent dimensions (Yadav et al., 2023). The resulting merge emphasizes parameters with consistent directional
    contributions.

  • DARE-TIES: Combining sparsification and sign-based conflict resolution, this variant first applies DARE-style pruning
    and then performs TIES-based trimming (Yadav et al., 2023; Yu et al., 2024b). The hybrid design aims to further
    suppress interference while preserving task-relevant updates.

  • DELLA: This method identifies task-relevant subspaces through low-rank decomposition and performs merging within
    these aligned representations (Deep et al., 2024). By operating in a shared latent space, it improves compatibility across
    merged models.

                                                              17

## Page 18

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

  • RankMean: Instead of averaging raw parameter values, RankMean aggregates parameters based on their relative
    rankings across models (Perin et al., 2024). This rank-based formulation increases robustness to scale mismatches and
    outlier updates.
  • CMA: CMA formulates model merging as a black-box optimization problem and applies covariance matrix adaptation
    to search for optimal merging coefficients (Akiba et al., 2025). The method iteratively updates a population of
    candidates based on validation performance.
  • AdaMerging: This approach learns adaptive, layer-wise merging coefficients guided by validation feedback (Yang
    et al., 2024). The coefficients are optimized to balance task contributions across different network depths.
  • Fisher: Fisher-weighted merging leverages Fisher information to estimate parameter importance for each
    model (Matena & Raffel, 2022). Parameters deemed more critical receive higher weights during aggregation, reducing
    harmful interference.
  • RegMean: By incorporating regularization terms during aggregation, this method penalizes deviations from important
    parameters of individual models (Jin et al., 2023). The regularized formulation stabilizes merging under heterogeneous
    tasks.
  • PSO-Merging: PSO-Merging formulates model merging as a particle swarm optimization problem in parameter
    space (Zhang et al., 2025). It iteratively updates candidate merged models based on personal and global best solutions
    guided by task performance, enabling data-driven yet gradient-free merging.
  • Model Swarm: Model Swarm treats each expert model as a particle and performs collaborative search in the weight
    space under a task-specific utility function (Feng et al., 2025). Instead of static aggregation, it adaptively explores
    model combinations and returns the best-performing expert discovered during optimization.

                                       Table 9. Details of task datasets used in our experiments.

                                                                                    Size
                Dataset       Source                                                                     Domain
                                                                              Val      Test
                CoLA           (Warstadt et al., 2019)                         50      1043                Misc.
                MNLI           (Williams et al., 2018)                         50      9815                Misc.
                MRPC           (Dolan & Brockett, 2005)                        50       408                News
                QNLI           (Levesque et al., 2011)                         50      5463              Wikipedia
                QQP6          /                                                50     40430         Social QA questions
                RTE             (Bar-Haim et al., 2006; Bentivogli et al.,     50      277           News, Wikipedia
                              2009; Dagan et al., 2006; Giampiccolo et al.,
                              2007)
                SST-2          (Socher et al., 2013)                           50      872            Movie reviews
                STS-B          (Cer et al., 2017)                              50     1500               Misc.
                MMLU          (Hendrycks et al., 2021)                        200     1000           Knowledge
                MMLU-Pro      (Wang et al., 2024)                              70     1000           Knowledge
                Hellaswag     (Zellers et al., 2019)                          200     1000           Knowledge
                K-Cross       (Ding et al., 2024)                             200     1000     Cross-domain reasoning
                GSM8k         (Cobbe et al., 2021)                            200     1000         Reasoning tasks
                NLGraph       (Wang et al., 2023; Zhang et al., 2024)         200     1000         Reasoning tasks
                TruthfulQA    (Lin et al., 2021)                              200     617      Safety-related evaluation
                AbstainQA     (Gehman et al., 2020)                           200     1000     Safety-related evaluation


C.2. Datasets
We use sixteen datasets from established benchmarks, covering natural language understanding, reasoning, and safety
evaluation. Eight datasets are drawn from the GLUE benchmark (Wang et al., 2018), including CoLA, MNLI, MRPC,
QNLI, QQP, RTE, SST-2, and STS-B, which together represent a broad spectrum of natural language understanding tasks.
In addition, we include eight widely used datasets for evaluating knowledge, reasoning, and safety evaluation, which are
MMLU (Hendrycks et al., 2021), MMLU-Pro (Wang et al., 2024), HellaSwag (Zellers et al., 2019), GSM8K (Cobbe et al.,
2021), K-Cross (K-Cross) (Ding et al., 2024), NLGraph (Wang et al., 2023; Zhang et al., 2024), TruthfulQA (Lin et al.,
2021), and AbstainQA (Gehman et al., 2020). Detailed dataset statistics are reported in Table 9.

                                                                  18

## Page 19

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization

C.3. Expert Models
Our experiments involve a total of eighteen expert models to be merged. All experts share the same architecture within each
setting and are independently adapted from a common base model using different training datasets. An overview of the
expert models and their corresponding training data is provided in Table 10. Eight experts are obtained by fine-tuning FLAN-
T5-base7 on individual GLUE datasets (Wang et al., 2018), while the remaining ten experts are adapted from Qwen2.5-1.5B8
using LoRA-based parameter-efficient fine-tuning on supervised fine-tuning datasets from the Tulu-v29 (Ivison et al., 2023)
collection. All experts are trained for five epochs with an initial learning rate of 2 × 10−4 and an effective batch size of 32.

                                           Table 10. Expert models used in our experiments.

                Base model     Source            Train Dataset                                          Models
                                                 CoLA (Warstadt et al., 2019)                       FLAN-T5-cola11
                                                 MNLI (Williams et al., 2018)                       FLAN-T5-mnli12
                                                 MRPC (Dolan & Brockett, 2005)                      FLAN-T5-mrpc13
                FLAN-T510      Open models       QNLI (Levesque et al., 2011)                       FLAN-T5-qnli14
                                                 QQP15                                              FLAN-T5-qqp16
                                                 RTE (Bar-Haim et al., 2006; Bentivogli et al.,      FLAN-T5-rte17
                                                 2009; Dagan et al., 2006; Giampiccolo et al.,
                                                 2007)
                                                 SST-2 (Socher et al., 2013)                        FLAN-T5-sst218
                                                 STS-B (Cer et al., 2017)                           FLAN-T5-stsb19
                                                 flan (Chung et al., 2024)                                  -
                                                 CoT (Wei et al., 2022)                                     -
                                                 OpenAssistant-1 (Köpf et al., 2023)                       -
                                                 ShareGPT 21                                                -
                QWEN2.5        LoRA-based        Code Alpaca (Chaudhary, 2023)                              -
                 -1.5B20        fine-tuning      LIMA (Zhou et al., 2023)                                   -
                                                 WizardLM Evol-Instruct v2 (Xu et al., 2024a)               -
                                                 Open-Orca (Lian et al., 2023)                              -
                                                 Science Literature (Ivison et al., 2023)                   -
                                                 Gemini Alpaca (distilled)                                  -


C.4. More Experimental Results

Table 11. Ablation study of EvoGM on 8 single-task benchmarks. Best results are highlighted with bold and a light-gray background,
computed independently for each column (Val/Test).

                         MMLU        MMLU-Pro HellaSwag            K-Cross      GSM8K         NLGraph           TruthQA    AbstainQA
       Method
                        Val   Test   Val      Test   Val   Test   Val    Test   Val   Test    Val    Test   Val     Test   Val   Test
   Single-Generator    0.620 0.576 0.385 0.232 0.650 0.590 0.425 0.412 0.540 0.339 0.515 0.529 0.515 0.439 0.210 0.119
     w/o Rounds        0.630 0.614 0.400 0.247 0.640 0.593 0.450 0.417 0.510 0.337 0.515 0.521 0.510 0.441 0.215 0.142
    w/o Cycle Loss     0.620 0.576 0.386 0.232 0.645 0.588 0.435 0.396 0.410 0.326 0.410 0.372 0.515 0.429 0.210 0.119
       EvoGM           0.640 0.625 0.429 0.224 0.660 0.594 0.450 0.379 0.495 0.434 0.545 0.537 0.540 0.441 0.230 0.121




   7
     https://huggingface.co/google/flan-t5-base
   8
     https://huggingface.co/Qwen/Qwen2.5-1.5B
   9
     https://huggingface.co/collections/allenai/tulu-v2-suite

                                                                  19

## Page 20

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization




                                                            GSM8K                                                 HellaSwag                                         Knowledge Crosswords                                                  MMLU
                               0.50                                                                                                                     0.43
                                                                                            0.66                                                                                                                0.64
                                                                                                                                                        0.42




         Top-5 Mean Accuracy
                               0.45                                                         0.64                                                                                                                0.63
                                                                                                                                                        0.41

                                                                                            0.62                                                        0.40                                                    0.62
                               0.40
                                                                                                                                                        0.39
                                                                                            0.60                                                                                                                0.61
                                                                                                                                                        0.38
                               0.35                                         EvoGM                                                  EvoGM                                                        EvoGM                                                   EvoGM
                                                                            PSO-Merging                                            PSO-Merging          0.37                                    PSO-Merging     0.60                                    PSO-Merging
                                                                                            0.58
                                                                            Model Swarm                                            Model Swarm                                                  Model Swarm                                             Model Swarm
                                                                                                                                                        0.36
                                       0              2                 4          6               0              2           4            6                    0           2              4            6               0           2             4             6
                                                  MMLU-Abstain                                                    MMLU-Pro                                                       NLGraph                                            TruthfulQA
                      0.250                                                                                                                             0.55                                                    0.54
                                                                                            0.42                                                                         EvoGM                                                   EvoGM
                                                                                                                                                                         PSO-Merging                                             PSO-Merging
                      0.225




Top-5 Mean Accuracy
                                                                                            0.40                                                        0.50             Model Swarm                                             Model Swarm
                                                                                                                                                                                                                0.52
                      0.200
                                                                                            0.38
                                                                                                                                                        0.45
                      0.175                                                                                                                                                                                     0.50
                                                                                            0.36
                      0.150                                                                                                                             0.40
                                                                                            0.34
                                                                                                                                                                                                                0.48
                      0.125                                                 EvoGM                                                  EvoGM                0.35
                                                                                            0.32
                                                                            PSO-Merging                                            PSO-Merging
                      0.100                                                 Model Swarm     0.30                                   Model Swarm
                                                                                                                                                        0.30                                                    0.46
                                       0              2                 4          6               0              2           4            6                    0           2              4            6               0           2             4             6
                                                            Iteration                                                 Iteration                                                   Iteration                                              Iteration

Figure 6. Fitness evolution of EvoGM and SOTA methods in multi-task scenarios. The curves represent the mean performance of the top
five individuals in the population, with shaded areas indicating the confidence intervals computed from these individuals.




                                                       MMLU                                                  MMLU-Pro                                                      HellaSwag                                                    K-Cross
                0.70
                                                                                           0.45                                                         0.70                                                    0.475

                0.65                                                                       0.40                                                                                                                 0.450
                                                                                                                                                        0.65
                                                                                                                                                                                                                0.425


Accuracy
                0.60                                                                       0.35
                                                                                                                                                        0.60                                                    0.400
                                                                                           0.30
                0.55                                                                                                                                                                                            0.375
                                                                                                                                                        0.55
                                                                                           0.25                                                                                                                 0.350
                0.50
                                                                                                                                                        0.50                                                    0.325
                                                                                           0.20

                                      Sing
                                           le               nds         Cycle    EvoG
                                                                                       M
                                                                                                   Sing
                                                                                                       le             nds         Cycle   EvoG
                                                                                                                                               M
                                                                                                                                                                Sing
                                                                                                                                                                    le              nds
                                                                                                                                                                                                Cycle   EvoG
                                                                                                                                                                                                            M
                                                                                                                                                                                                                        Sing
                                                                                                                                                                                                                            le              nds
                                                                                                                                                                                                                                                        Cycle   EvoG
                                                                                                                                                                                                                                                                    M
                                                      Rou         w/o                                             Rou       w/o                                                 Rou       w/o                                           Rou       w/o
                                                w/o                                                         w/o                                                           w/o                                                     w/o
                                                      GSM8K                                                   NLGraph                                                           TruthQA                                             AbstainQA
                                                                                                                                                                                                                 0.24
                0.55
                                                                                           0.55                                                         0.55
                                                                                                                                                                                                                 0.22
                0.50
                                                                                           0.50                                                                                                                  0.20



Accuracy
                                                                                                                                                        0.50
                0.45
                                                                                                                                                                                                                 0.18
                                                                                           0.45
                0.40                                                                                                                                    0.45                                                     0.16
                                                                                           0.40
                0.35                                                                                                                                                                                             0.14
                                                                                                                                                        0.40
                                                                                           0.35                                                                                                                  0.12
                0.30
                                                                                           0.30                                                         0.35                                                     0.10
                                      Sing
                                           le               nds         Cycle    EvoG
                                                                                       M
                                                                                                   Sing
                                                                                                       le             nds         Cycle   EvoG
                                                                                                                                               M
                                                                                                                                                                Sing
                                                                                                                                                                    le              nds
                                                                                                                                                                                                Cycle   EvoG
                                                                                                                                                                                                            M
                                                                                                                                                                                                                        Sing
                                                                                                                                                                                                                            le              nds
                                                                                                                                                                                                                                                        Cycle   EvoG
                                                                                                                                                                                                                                                                    M
                                                      Rou         w/o                                             Rou       w/o                                                 Rou       w/o                                           Rou       w/o
                                                w/o                                                         w/o                                                           w/o                                                     w/o

                                                                                                                                           Val                 Test


Figure 7. PAblation study on different components. (1) Single-Generator represents the single generative model variant; (2) w/o Rounds
represents the replacement of multi-round updates with five continuous iterations; and (3) w/o Cycle Loss represents the removal of the
cycle-consistency constraint.




                                                                                                                                                   20

## Page 21

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization




Table 12. Ablation study on key hyperparameters. The baseline configuration is marked with “–”. In the ∆ Test Acc column, the largest
improvement is highlighted in bold, while performance degradation is shown in italic.

             Parameter                  Value     Pop Iter Epochs          LR      Rounds Dev Acc Test Acc ∆ Test Acc
                                          10      10     5     400        0.001      3       0.4954      0.4100      +0.0050
             Population Size              20      20     5     400        0.001      3       0.5104      0.4180         –
                                          30      30     5     400        0.001      3       0.5129      0.4240      +0.0305
                                          3       20     3     400        0.001      3       0.5200      0.4195      +0.0145
             Max Iterations               5       20     5     400        0.001      3       0.5104      0.4180         –
                                          8       20     8     400        0.001      3       0.5150      0.4245      +0.0195
                                         200      20     5     200        0.001      3       0.5125      0.4245      +0.0115
             Generator Epochs            400      20     5     400        0.001      3       0.5104      0.4180         –
                                         600      20     5     600        0.001      3       0.5129      0.4240      -0.0035
                                       0.0001     20     5     400        0.0001     3       0.5175      0.4270      +0.0220
             Learning Rate             0.0010     20     5     400        0.0010     3       0.5104      0.4180         –
                                       0.0050     20     5     400        0.0050     3       0.5175      0.4155      +0.0105
                                          1       20     5     400        0.001      1       0.5046      0.4015      -0.0165
             Evolution Rounds             3       20     5     400        0.001      3       0.5104      0.4180         –
                                          5       20     5     400        0.001      5       0.5264      0.4195      +0.0130




                      Table 13. Performance comparison across different numbers of models and evaluation stages.

  Models                           Initial Val                            Models                           Final Val
             MMLU MMLU-Pro HellaSwag K-Cross                   Avg                   MMLU MMLU-Pro HellaSwag K-Cross               Avg
      2       0.580        0.329         0.600         0.390   0.475         2       0.575       0.329            0.605    0.395   0.476
      3       0.620        0.300         0.625         0.360   0.476         3       0.615       0.314            0.640    0.365   0.484
      4       0.585        0.314         0.635         0.375   0.477         4       0.605       0.400            0.645    0.345   0.499
      5       0.615        0.314         0.615         0.355   0.475         5       0.605       0.414            0.650    0.350   0.505
      6       0.605        0.286         0.615         0.360   0.466         6       0.615       0.400            0.660    0.345   0.505
      7       0.605        0.286         0.595         0.370   0.464         7       0.595       0.400            0.655    0.395   0.511
      8       0.600        0.386         0.565         0.360   0.478         8       0.630       0.414            0.645    0.345   0.509
      9       0.600        0.386         0.565         0.360   0.478         9       0.615       0.386            0.655    0.375   0.508
     10       0.600        0.386         0.565         0.360   0.478        10       0.620       0.343            0.650    0.385   0.499
  Models                           Initial Test                           Models                          Final Test
             MMLU MMLU-Pro HellaSwag K-Cross                   Avg                   MMLU MMLU-Pro HellaSwag K-Cross               Avg
      2       0.559        0.196         0.578         0.400   0.433         2       0.552       0.193            0.571    0.400   0.429
      3       0.576        0.255         0.563         0.385   0.445         3       0.578       0.254            0.561    0.385   0.445
      4       0.587        0.241         0.578         0.385   0.448         4       0.601       0.250            0.590    0.386   0.457
      5       0.579        0.251         0.585         0.391   0.452         5       0.600       0.253            0.601    0.380   0.459
      6       0.577        0.235         0.593         0.388   0.448         6       0.602       0.245            0.589    0.391   0.457
      7       0.574        0.234         0.592         0.394   0.449         7       0.589       0.234            0.599    0.393   0.454
      8       0.582        0.232         0.575         0.344   0.433         8       0.608       0.246            0.599    0.386   0.460
      9       0.582        0.232         0.575         0.344   0.433         9       0.595       0.244            0.595    0.398   0.458
     10       0.582        0.232         0.575         0.344   0.433        10       0.582       0.228            0.590    0.405   0.451




                                                                     21

## Page 22

EvoGM: Learning to Merge LLMs via Evolutionary Generative Optimization




Figure 8. Performance Comparison across Multiple Benchmarks. The radar plots illustrate the performance of different model merging
methods on Validation (top row) and Test (bottom row) sets. The benchmarks include HellaSwag, Knowledge Crosswords, MMLU Pro,
and MMLU, with the rightmost column showing the overall average performance across all tasks.




                                                               22
