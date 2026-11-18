# Extracted pages: 2603.01712.pdf

## Page 1

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents


                                               Qizheng Li 1 2 * Yifei Zhang 1 3 * Xiao Yang 1 * Xu Yang 1 * Zhuo Wang 1 4 Weiqing Liu 1 † Jiang Bian 1


                                                                   Abstract                                      While prompt engineering offers a lightweight entry point,
                                               Fine-tuning large language models for vertical                    production-grade applications typically require fine-tuning
                                               domains remains labor-intensive, requiring practi-                to meet reliability and performance standards (Zhang et al.,
                                               tioners to curate data, configure training, and itera-            2023; Huang et al., 2024). However, fine-tuning remains a




arXiv:2603.01712v2 [cs.AI] 20 May 2026
                                               tively diagnose model behavior. Despite growing                   labor-intensive process that must be re-engineered for each
                                               interest in autonomous machine learning and lan-                  new domain: domain experts and ML engineers must col-
                                               guage agents, end-to-end LLM fine-tuning has                      laborate to interpret requirements, curate unstructured data,
                                               not been systematically studied as an interactive                 configure training pipelines, and rigorously evaluate model
                                               agent task. We introduce FT-Dojo, an interactive                  checkpoints. A natural next frontier emerges: can agents
                                               benchmark environment for autonomous LLM                          automate the end-to-end LLM fine-tuning process?
                                               fine-tuning, comprising 13 tasks across 5 domains.                Recent advances in AI-for-AI illustrate a steady broadening
                                               Rather than a new collection of static datasets,                  of agent capabilities, paving the way to automate the labor-
                                               FT-Dojo standardizes a task interface, shared                     intensive work involved in AI research and development.
                                               raw-data repository, sandboxed execution envi-                    Benchmarks such as MLE-Bench (Chan et al., 2024) and
                                               ronment, structured feedback protocol, and held-                  DSBench (Jing et al., 2024) evaluate agents on tasks in-
                                               out evaluation procedure. We further develop FT-                  volving data preprocessing, feature engineering, and model
                                               Agent, a fine-tuning-oriented autonomous frame-                   training, quantitatively measuring how well an agent auto-
                                               work that uses structured iteration planning, fail-               mates the end-to-end Machine Learning Engineering (MLE)
                                               fast validation, and multi-level feedback analy-                  process. However, end-to-end LLM fine-tuning is substan-
                                               sis to refine data and training strategies. Ex-                   tially more open-ended and complex, compared with tradi-
                                               periments show that FT-Agent provides a strong                    tional MLE: (i) Data must be constructed, not merely pro-
                                               initial baseline, achieving the best performance                  cessed: practitioners navigate heterogeneous raw sources,
                                               on 10 out of 13 tasks, with additional controlled                 leveraging diverse tools such as LLM APIs for synthesis
                                               comparisons against frontier agents, open-source                  and domain-specific processors for cleaning; (ii) Analysis
                                               planning backbones, and multi-run statistics sup-                 signals extend beyond aggregate metrics: practitioners in-
                                               porting the main findings. Case studies show                      terpret training dynamics, validation curves, and sampled
                                               that agents can recover from failures through                     model outputs to diagnose capability gaps. The iterative
                                               cumulative learning, while still exposing limita-                 joint optimization over data construction and LLM finetun-
                                               tions in causal diagnosis and long-horizon plan-                  ing over various analysis signals has neither been formally
                                               ning. The implementation is available at https:                   defined as an agent task, nor systematically evaluated.
                                               //github.com/microsoft/rd-agent.
                                                                                                                 Motivated by these challenges, we introduce FT-Dojo,
                                                                                                                 the first interactive benchmark environment for evaluating
                                         1. Introduction                                                         agents on LLM fine-tuning. The open-ended, iterative re-
                                                                                                                 finement inherent to agentic LLM finetuning necessitates
                                         Large language models have achieved remarkable general                  a sandboxed interactive environment in which agents can
                                         capabilities (Liu et al., 2025; Team et al., 2025), yet de-             repeatedly refine their approach based on evaluation feed-
                                         ploying them in specialized domains such as chemistry, le-              back. To approximate real-world end-to-end LLM finetun-
                                         gal, and finance often demands domain-specific adaptation.              ing, FT-Dojo aggregates diverse raw data sources into a
                                           *
                                             Equal contribution 1 Microsoft Research Asia 2 Peking Univer-       shared repository: agents must autonomously identify rele-
                                         sity 3 Nanjing University 4 The University of Chicago. Correspon-       vant sources, decide how to filter and transform them into
                                         dence to: Weiqing Liu <Weiqing.Liu@microsoft.com>.                      training instances, and leverage LLM APIs and other tools
                                                                                                                 for data synthesis and cleaning. The contribution is there-
                                         Proceedings of the 43 rd International Conference on Machine            fore not a new collection of static datasets; rather, FT-Dojo
                                         Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026
                                         by the author(s).                                                       standardizes an interactive protocol, execution environment,

                                                                                                             1

## Page 2

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

feedback interface, and held-out evaluation procedure for           against third-party frontier agent systems and planning back-
autonomous fine-tuning. This design enables both data               bones under the controlled FT-Dojo protocol.
construction and finetuning configuration to be first-class
optimization targets. FT-Dojo comprises 13 tasks spanning           2. FT-Dojo
5 domains (Math, Patent Examination, Chemistry, Finance,
and Table QA).                                                      2.1. Task Definition
In FT-Dojo, we observe that general-purpose agents such             Given a task specification τ describing the target capabil-
as OpenHands (Wang et al., 2024) perform poorly even                ity, a data repository D containing heterogeneous raw data
when supported by strong LLM backends and the same                  sources, and an evaluation function F, an agent must au-
fine-tuning tools. Our analysis shows that general-purpose          tonomously produce a fine-tuned model M by: (1) selecting
agents often overlook key signals in noisy feedback, leading        and processing a subset d ⊆ D into training instances, (2)
to low-value refinements and significant resource waste—an          configuring training hyperparameters θt , and (3) iteratively
especially acute issue in LLM-finetuning scenarios. Moti-           refining based on evaluation feedback f (t) .
vated by this gap, we design FT-Agent to better align with
                                                                    Formally, let θd denote the data strategy (source selection,
the fine-tuning scenario. FT-Agent draws on expert practice
                                                                    filtering, formatting), θt the training hyperparameters, and
and introduces targeted mechanisms to address the main
                                                                    f (t) = Fτ (M (t) ) the structured feedback returned by the
obstacles we observe in FT-Dojo: (i) structured iteration
                                                                    evaluator at iteration t (with f (0) = ∅ for the initial itera-
planning that keeps growing context manageable and helps
                                                                    tion). The agent’s objective is:
the agent focus on high-level hypotheses instead of verbose
logs, (ii) fail-fast validation that prevents wasted compu-
                                                                        θd∗ , θt∗ = arg max Fτ Train(θd (D), θt ; f (t−1) )
                                                                                                                            
                                                                                                                                (1)
tation by catching configuration and data problems early,                            θd ,θt
and (iii) structured feedback analysis that helps the agent
interpret metrics, loss curves, and error cases to decide how       where Fτ is the task-specific evaluator that assesses model
to refine data and training strategies. Through this design,        performance on the held-out benchmark associated with
FT-Agent delivers stable, efficient, and meaningful analysis        task specification τ (accessed via the Meta API I), θd (D)
signals across iterations in FT-Dojo, enabling the backend          denotes the training data obtained by applying strategy θd
LLM to generate deeper and more insightful refinement               to the data repository D, and Train(·; f (t−1) ) executes
proposals.                                                          fine-tuning informed by previous feedback and returns the
                                                                    resulting model.
Our contributions are threefold: (i) We introduce FT-Dojo,
the first interactive benchmark environment for end-to-end          2.2. System Architecture
LLM fine-tuning, where agents must autonomously navi-
gate from raw data sources to trained models across 13 tasks        FT-Dojo adopts a modular architecture where components
in 5 domains, with both data strategy and training configu-         operate independently through well-defined interfaces, as
ration as optimization targets; (ii) We propose FT-Agent, a         illustrated in Figure 1. This design enables flexible task
fine-tuning-oriented agent framework that introduces three          extension and ensures reproducible evaluation.
targeted mechanisms described above. These mechanisms               Each task executes within a sandboxed Docker container
help the agent avoid basic engineering pitfalls, focus on           that ensures reproducibility and security. The sandbox
high-level decisions, and better reveal algorithmic capa-           provides: (1) workspace isolation with separated private
bilities of the underlying LLM, providing a strong initial          storage; (2) configurable resource constraints including
baseline for the benchmark. (iii) Experiments show FT-A-            GPU/CPU limits and execution timeouts; and (3) secure
gent achieves the best performance on 10 of 13 tasks, and           access to external LLM APIs for data synthesis. Within this
additional frontier-agent baselines, open-source backbone           environment, agents interact with three core interfaces:
variants, and 3-run statistics support the main qualitative
findings. Case analyses reveal emergent capabilities (failure       Meta API (I). Exposes task specifications, system capa-
recovery through historical memory, iterative self-improve-         bilities, data descriptions, and API documentation through
ment) and fundamental limitations in causal reasoning, high-        a unified interface, enabling programmatic access to task
lighting the promise and boundaries of autonomous LLM               requirements, available resources, and permitted operations.
fine-tuning.                                                        Data Repository (D). A shared pool of heterogeneous
                                                                    data sources aggregated across all benchmark tasks. Un-
Conflict of Interest Disclosure. Several authors are em-            like prior benchmarks that provide fixed datasets per task,
ployed by Microsoft Research Asia, which developed the              FT-Dojo creates an open decision space where data selec-
FT-Agent system and the R&D-Agent codebase evaluated                tion, filtering, and processing become integral parts of the
and released in this paper. The experiments also compare            optimization problem.

                                                                2

## Page 3

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents




Figure 1. Overview of FT-Dojo architecture and agent interaction workflow. Agents operate within an isolated sandbox and interact with
three core interfaces: Meta API I for task and system information, Data Repository D for training data, and Evaluator F that returns
structured feedback. Agents iteratively query information, submit code and configurations, execute training, and refine based on evaluation
feedback until achieving satisfactory performance.


Evaluator (F). Adapted from OpenCompass (Contributors,                  domain is curated from diverse sources, including domain-
2023), the evaluator assesses submitted models on held-                 specific corpora and benchmark-associated datasets. Eval-
out data using task-specific metrics. It returns structured             uation sets are drawn from established benchmarks, with
feedback comprising: (1) aggregate metric scores, (2) per-              strict separation from training data (details in Appendix A).
instance predictions with correctness labels, and (3) training
                                                                        Benchmark Framing. FT-Dojo is not intended as a claim
loss trajectories, enabling fine-grained diagnosis of failure
                                                                        that all constituent static datasets are newly introduced. The
modes.
                                                                        benchmarked object is the autonomous fine-tuning process
                                                                        under a unified interactive protocol. FT-Dojo standardizes
2.3. Environment Formulation                                            the task interface, shared raw data repository, sandboxed
FT-Dojo models autonomous LLM fine-tuning as an itera-                  execution environment, validation/test evaluation protocol,
tive agent-environment interaction. At each iteration, the              and structured feedback returned after each iteration. Ex-
agent observes the task specification τ , data repository D,            isting benchmarks serve as held-out evaluators for domain-
and evaluation feedback from prior submissions, then se-                specific target capabilities, while FT-Dojo evaluates whether
lects actions from four core operations: querying task and re-          an agent can construct data, configure training, run fine-
source metadata, processing raw data into training instances,           tuning, and improve from feedback.
executing fine-tuning pipelines, and submitting models for              Evaluation Protocol. FT-Dojo employs a two-phase evalua-
evaluation. After each submission, structured feedback from             tion protocol. After context-length filtering, each benchmark
the evaluator F enables the agent to diagnose failures, adjust          or subtask is partitioned into non-overlapping validation and
its strategy, and iterate.                                              test splits with Nval = Ntest = min(100, ⌊n/2⌋), where n
                                                                        is the number of filtered examples. Before final assessment,
2.4. Task Suite and Evaluation                                          agents can submit models for validation-set evaluation, re-
Each task in FT-Dojo is specified through two components:               ceiving structured feedback to support iterative refinement.
(1) a task objective, which is a natural language description           The test set remains inaccessible during optimization; for fi-
specifying the domain, target capability, and expected output           nal assessment, agents submit their best validation-selected
format; and (2) an evaluation metric, which is a script that            model for held-out test evaluation. The training-data cap
measures model performance on held-out instances using                  is enforced as a shared resource budget: each method may
domain-appropriate metrics.                                             select, filter, or synthesize up to the allowed number of train-
                                                                        ing instances using its own procedure, rather than training
Task Suite. FT-Dojo comprises 13 tasks spanning five do-                from a single precomputed subset.
mains, as summarized in Table 1. These domains were
selected to cover diverse fine-tuning challenges: scientific            Extensibility. To support flexible extension, both task ob-
reasoning (Math, Chemistry), domain-specific language un-               jectives and evaluation metrics are designed as pluggable
derstanding (Patent Examination, Finance), and structured               modules with unified interfaces. Task objectives are regis-
data comprehension (Table QA). Training data for each                   tered through the Meta API I, while evaluation metrics are


                                                                    3

## Page 4

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Table 1. Task suite in FT-Dojo spanning 13 tasks across 5 domains. Data Quality column describes training data characteristics relevant
for fine-tuning. Valid/Test indicates the size of both validation and test sets (identical for each task). See Appendix A for metric definitions
and abbreviations.

  Domain        Task                       Description                                         Train Data Quality Metrics            Train    Valid/Test
                                           Competition-level mathematical reasoning re-
                                           quiring multi-step symbolic derivations, alge-      82% lack solution
  Mathematics AIME 2025                                                                            No CoT          Accuracy          40315       15
                                           braic manipulation, and exact numerical an-
                                           swers
                                           Given a patent claim and candidate patents,
                Prior Art Retrieval        identify which were cited as prior art for re-          No CoT          Acc, Macro-F1     54028       100
  Patent                                   jecting the claim                                    Q&A pairs only
  Examination
                                           Classify patent claims as allowed, §102 rejec-       High complexity
                Novelty Classification                                                                             Acc, Macro-F1     136211      100
                                           tion (lacks novelty), or §103 rejection (obvious)
                                           Identify the most relevant prior art paragraph
                Paragraph Identification                                                                           Accuracy          64210       100
                                           from candidates that supports claim rejection
                                        Predict molecular properties from SMILES:
                Molecular Understanding functional group counting, scaffold extraction,                            MAE, Tanimoto      6319       160
                                        equivalence                                             Structured CoT
  Chemistry                                                                                     Expert validated
                                        Modify molecular structures by adding, delet-
                Molecule Editing                                                                                   Acc, Tanimoto      4497       50
                                        ing, or substituting functional groups
                                        Optimize molecules for target properties (bind-
                Molecule Optimization   ing affinity, solubility) while preserving scaf-                           SR, Valid Ratio    5587       300
                                        folds
                                        Predict reaction products, recommend condi-
                Reaction Prediction                                                                                FTS, BLEU         6820        237
                                        tions, and select plausible reaction mechanisms
                                           Answer Chinese financial certification exam
                                                                                                   No CoT
  Finance       Financial QA               questions requiring quantitative analysis and          MCQ format       Accuracy           6179       100
                                           domain reasoning
                                           Perform correlation analysis, trend forecasting,
                Data Analysis                                                                                   Accuracy              7372       170
                                           and statistical reasoning over tabular data
                                                                                                With CoT
  Table QA                                 Verify factual statements against tabular evi- Text & code reasoning
                Fact Checking              dence through value lookups and boolean veri-                        Exact Match           2282       48
                                           fication
                                           Perform arithmetic operations (sum, average,
                Numerical Reasoning                                                                             Accuracy              8892       197
                                           percentage, comparison) over table data
                                           Generate Python code to create specified charts
                Visualization                                                                                   Pass@1, ECR          1115        25
                                           and visualizations from table data



implemented as standalone scripts following a standardized                       interpretation required for end-to-end fine-tuning. In FT-
protocol: each script receives model predictions and ground-                     Dojo, we thoroughly evaluated general-purpose agents such
truth labels as input, and returns structured feedback through                   as OpenHands and found that they still perform poorly even
the Evaluator F. Despite varying metrics across domains                          when powered by the strongest LLM backends (Table 3).
(e.g., accuracy for classification, ROUGE for generation,                        In some tasks, their performance even falls below that of
Tanimoto similarity for molecular tasks), this unified inter-                    the original model without any fine-tuning. These system-
face enables users to extend FT-Dojo with custom tasks by                        atic observations suggest that general-purpose agents are
simply providing a task description and evaluation script,                       not well aligned with the following three challenges in au-
without modifying the core infrastructure.                                       tonomous LLM fine-tuning, which collectively undermine
                                                                                 the feasibility of autonomous LLM fine-tuning.
3. FT-Agent                                                                      Challenge 1: Long and ever-growing context over-
                                                                                 whelms the agent and hides future effective directions.
3.1. Challenges in Autonomous LLM Fine-Tuning
                                                                                 Fine-tuning produces a large and varied stream of artifacts,
There are currently no open-source agents designed specifi-                      including verbose training logs, model checkpoints, loss
cally for large-scale model post-training. Existing systems                      curves, and evaluation results. With multiple rounds of
such as OpenHands (Wang et al., 2024) are built for general-                     iteration, these artifacts accumulate quickly, and the con-
purpose code execution and automation, rather than the                           text grows to a scale that becomes difficult for an agent to
high-complexity decision-making and multimodal feedback                          manage. This hinders the agent from making meaningful


                                                                             4

## Page 5

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

decisions for future iterations.                                            prior iterations, structured evaluation signals, and the current
                                                                            system state, the agent formulates a unified hypothesis (see
Challenge 2: Inefficient Exploration. End-to-end fine-
                                                                            Appendix G.1) that jointly covers high-level data strategy
tuning is computationally expensive, making efficient iter-
                                                                            and high-level training configuration. Each iteration pro-
ation essential. Existing agents, however, lack such fail-
                                                                            duces a compact plan with three parts: (i) a data strategy de-
fast capabilities. They tend to execute full-scale training
                                                                            scribing source selection, filtering rules, formatting choices,
even when the configuration is malformed, data processing
                                                                            or synthetic augmentation; (ii) a training configuration cov-
scripts are buggy, or the dataset is incompatible with the
                                                                            ering learning-rate settings, batch-size scale, prompt-format
task format. This leads to wasted computation, significantly
                                                                            adjustments, CoT usage, or iteration length; and (iii) a ra-
fewer effective iterations, and poor overall performance in
                                                                            tionale that links the proposed changes to observed failure
FT-Dojo.
                                                                            patterns.
Challenge 3: Limited Understanding of Evaluation Feed-
                                                                            This unified hypothesis is generated through an experience-
back. Evaluation in fine-tuning is multifaceted: aggregate
                                                                            driven process: the agent inherits the current best-
metrics show overall ability, per-instance outputs expose
                                                                            performance configuration as the baseline, reviews sibling
consistent reasoning errors, and loss curves reflect optimiza-
                                                                            attempts to understand what has already been explored, and
tion behavior. Generic agents often fail to learn from this
                                                                            consults past failures to avoid repeating ineffective solu-
mixed and unstructured feedback, causing them to miss key
                                                                            tions. Formally, given the best configuration (θd∗ , θt∗ ) and
diagnostic signals and gain too little useful information even
                                                                            experience E,
after costly iterations.
                                                                                      (t+1)      (t+1)
                                                                                                         ) = Propose τ, (θd∗ , θt∗ ), E
                                                                                                                                          
To obtain more meaningful benchmark results, we aim to                             (θd        , θt                                            (2)
evaluate LLMs at the algorithmic level instead of letting
them get stuck on engineering issues. We introduce an agent                 By enforcing high-level descriptions instead of accumulat-
framework called FT-Agent, which draws on expert practice                   ing low-level operational details, FT-Agent keeps iteration
and provides core mechanisms that help better assess the                    histories manageable. These structured summaries support
backend LLM.                                                                both forward planning and backward review, helping the
                                                                            agent focus on conceptual iteration hypotheses without be-
3.2. FT-Agent Framework                                                     ing overwhelmed by growing fine-grained logs.
                                                                            Stage 2: Implementation & Fail-Fast Validation. To
                                                                            address the inefficiencies of generic agents, FT-Agent ap-
                                                                            plies progressive validation through three increasingly costly
                                                                            stages before full training is allowed to run: (i) static and
                                                                            schema validation, which performs syntax checking, depen-
                                                                            dency analysis, path verification, and detection of configura-
                                                                            tion errors; (ii) data format checks and mini-run validation,
                                                                            which run on a small sample or reduced-epoch test to verify
                                                                            output structure and catch broken data pipelines, unstable
                                                                            losses, or incompatible formats; and (iii) runtime sanity de-
                                                                            tection, which identifies anomalies such as exploding loss,
                                                                            empty datasets, or invalid gradients during a short mini-
                                                                            batch run.
                                                                            Configurations that fail any stage are rejected immediately,
                                                                            and the agent receives targeted diagnostic logs that encour-
Figure 2. Overview of FT-Agent. The agent iteratively explores              age rapid correction. This fail-fast process catches most er-
configurations, learning from failed trials (red) to reach successful       rors before significant computation is spent, enabling much
ones (green). Each iteration follows three stages: Strategy Pro-            higher iteration throughput. As seen in §4.5, FT-Agent com-
posal, Implementation & Validation, and Feedback Analysis.                  pletes 7 loops within 10 hours on Fact Checking, while
                                                                            OpenHands issues 410 conversation events in a single it-
FT-Agent operates as a three-stage loop executed at every                   eration before timing out. Once all validations pass, the
iteration, where each stage directly addresses one challenge                full pipeline runs, including complete data curation θd (D),
outlined above.                                                             model fine-tuning Train(·), and evaluation on the valida-
                                                                            tion set.
Stage 1: Strategy Proposal. Given the task specification
retrieved through the Meta API I(τ ), distilled summaries of                Stage 3: Structured Feedback Aggregation and Expert-

                                                                        5

## Page 6

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Guided Diagnosis. After a successful training run, the                 icy under the same external tools. This factorization is
evaluator returns multi-level structured feedback, including:          intended to isolate the effect of fine-tuning-oriented agent
(i) overall domain metrics, (ii) per-instance predictions with         design from differences in tool availability or runtime con-
error tags, (iii) loss curves showing training dynamics, and           ditions.
(iv) sampled failure cases that reveal recurring error patterns.
                                                                       Model Configuration. FT-Agent utilizes GPT-5.2 as the in-
FT-Agent analyzes these signals to identify capability gaps,           teractive backbone. For data processing, we employ GPT-5
signs of overfitting or underfitting, domain mismatch, noise           and GPT-4o-mini specifically for data cleaning and synthe-
sensitivity, and systematic reasoning problems such as miss-           sis. For evaluations that require an LLM to judge correctness
ing CoT structure, table parsing mistakes, or molecule-level           against the ground truth, we use GPT-5 as the judge model
errors. This diagnosis supports coordinated refinement: the            with a fixed rubric and identical prompts across all methods
agent can expand data to cover failed patterns while also              (Appendix A.4).
adjusting training settings such as learning rate or training
                                                                       Baselines. We compare FT-Agent against three dis-
duration. If the current run outperforms the previous best,
                                                                       tinct paradigms: (1) Base Model: The un-tuned
FT-Agent updates the best configuration,
                                                                       Qwen2.5-7B-Instruct model. (2) Manual SFT: A
       (θd∗ , θt∗ ) ← (θd , θt )   if   Fτ (M ) > Fτ (M ∗ )            constrained, resource-matched manual baseline executed
                                                                       by senior NLP practitioners using rule-based cleaning and
The full experiment and its feedback are recorded as experi-           manual tuning without LLM assistance. This baseline is not
ence for later iterations. Through this feedback-driven loop,          intended as an expert upper bound; Appendix B describes
the agent forms clear hypotheses about what failed and how             its two-pass constraint and an LLM-assisted variant. (3)
to improve next, enabling steady, expert-like progress across          Tool-Augmented OpenHands: A strong baseline based on
iterations.                                                            OpenHands (Wang et al., 2024). Crucially, we equipped
                                                                       this agent with the identical fine-tuning primitives (Lla-
                                                                       maFactory wrappers) and runtime environment as FT-Agent.
4. Experiments                                                         (detailed description in Appendix B).
4.1. Experimental Setup
                                                                       4.2. Main Results
We evaluate on the 13 tasks across 5 domains in the FT-
Dojo suite (Table 1). Each experiment is conducted on a                Table 2 presents the comparative evaluation of FT-Agent
single NVIDIA B200 GPU under a strict 12-hour end-to-                  against three baselines across 13 tasks. Overall, FT-Agent
end wall-clock budget, which includes agent interaction,               achieves the best performance on 10 out of 13 tasks, es-
data cleaning when applicable, fine-tuning, validation, and            tablishing a strong initial autonomous baseline for FT-Dojo.
final evaluation. Qwen2.5-7B-Instruct (Qwen et al., 2025)              The multi-run results show that the main qualitative conclu-
serves as the target base model. To simulate data-scarce               sions hold despite stochastic variation, with larger variance
scenarios and ensure fair comparison, we restrict training to          on chemistry reaction similarity metrics.
standard Supervised Fine-Tuning (SFT) or LoRA, capped                  Breakthroughs on Sparse Supervision. The most striking
at a maximum of 2,000 training samples per task. This                  result is observed on the AIME 2025 task. As detailed
cap is a per-method resource budget: each method inde-                 in Table 1, 82% of raw training samples lack solutions
pendently selects, filters, or synthesizes its own training            and provide no Chain-of-Thought (CoT) rationale. Conse-
instances under the same limit. To account for variance                quently, the base model, tool-augmented OpenHands, and
in the agent’s stochastic optimization process, we run FT-             constrained Manual SFT baseline all achieve 0% accuracy.
Agent three times with independent initializations and report          FT-Agent reaches 11.11% on average across three runs,
mean ± standard deviation across runs. Running 3 seeds                 indicating that the agent can identify missing reasoning
for all methods and all 13 tasks would require over 1,500              supervision as a bottleneck and synthesize useful training
GPU-hours, so we report multi-run statistics for FT-Agent              trajectories under the fixed resource budget.
and single controlled runs for the comparison baselines.
                                                                       Proficiency in Code and Structure. Beyond pure reason-
Controlled Protocol. Across autonomous-agent systems,                  ing, FT-Agent is strongest on tasks requiring executable
we hold fixed the external resources and runtime conditions:           code generation and strict structural constraints. In Ta-
the same target base model, raw data repository, Docker                ble QA Visualization, FT-Agent achieves a Pass@1 of
sandbox, LlamaFactory/OpenCompass wrappers, LLM API                    29.33%, outperforming the tool-augmented OpenHands
access, GPU hardware, validation/test split, and 12-hour               baseline (24.00%). Similarly, in Chemistry Molecule Edit-
wall-clock budget. What varies is the agent framework: FT-             ing, it attains 54.44% accuracy, outperforming OpenHands
Agent uses fine-tuning-oriented orchestration, while Open-             (40.00%). Manual SFT remains competitive in standard-
Hands retains a general-purpose CodeAct-style control pol-

                                                                   6

## Page 7

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Table 2. Main results comparison. We compare FT-Agent against baselines across 13 tasks in 5 domains. FT-Agent achieves the best
performance on 10 out of 13 tasks. Bold indicates best performance. FT-Agent results are reported as mean±std over 3 independent runs.
Manual SFT is a constrained, resource-matched manual baseline without LLM assistance. See Appendix A for metric definitions and
abbreviations.
                       Math           Patent Examination                         Table QA                     Finance                                        Chemistry
                       AIME     PAR4PC     NOC4PC       PI4PC      DA          FC        NR          Vis.      QA                 Mol Und              Mol Edit       Mol Opt                Reaction
Method                 Acc↑       F1↑        F1↑        Acc↑       Acc↑       Acc↑       Acc↑       P@1↑       Acc↑      MAE↓     TMS↑        Acc↑      Acc↑        SR↑     VS↑           FTS↑      Acc↑
Qwen2.5-7B-Instruct    0.00      57.78      26.20      34.00      32.78      75.00      32.00       4.00      65.10      0.53      0.09      63.67      22.20      23.33      68.67      15.59       30.00
Manual SFT             0.00      67.61      33.39      28.00      28.80      70.83      43.00       8.00      71.68      0.65      0.00      36.67       0.00       9.67      31.00       0.00       10.00
OpenHands              0.00      50.51      28.44      44.00      23.22      66.67      40.00      24.00      64.73      0.65      0.10      68.00      40.00      31.00      80.67      35.59       60.00
FT-Agent            11.11±3.82 67.61±2.40 36.81±4.13 49.00±4.36 34.71±0.28 81.25±2.08 45.33±1.15 29.33±6.11 67.20±0.31 0.23±0.16 0.36±0.06 67.55±5.68 54.44±2.94 34.44±2.17 90.00±4.41 30.92±11.22 35.33±3.06




ized formats such as Financial QA, which suggests that au-                                                  strained manual comparison rather than to replace the full
tonomous fine-tuning is most useful when the task requires                                                  13-task baseline. Open-source FT-Agent variants remain
iterative data diagnosis, dynamic tool use, or constraint-                                                  competitive, with DeepSeek-V3.2 surpassing OpenHands
aware optimization. Table 4 includes the LLM-assisted                                                       and approaching Codex on average. These results support
Manual SFT variant as a five-task reference.                                                                FT-Agent’s role as a strong fine-tuning-oriented baseline
                                                                                                            rather than establishing it as a universally superior agent
Table 3. Exploration dynamics between OpenHands and FT-Agent.                                               framework.

  Metric                                          OpenHands                  FT-Agent                       Budget Sensitivity. To test whether the 12-hour wall-clock
                                                                                                            budget disproportionately favors fail-fast designs, we also
  Avg. Loops                                            3.69                    8.77
                                                                                                            ran OpenHands on AIME 2025 with a 24-hour budget across
  Avg. Improve Rate                                    17.31%                  24.37%
  Avg. Cost (iterative only)                            $3.92                   $5.73                       two runs. We observed no meaningful improvement over the
                                                                                                            12-hour setting: in both runs, the best checkpoint appeared
                                                                                                            within the first 6 hours, and later iterations did not improve
Exploration Depth and Efficiency. Table 3 shows that FT-                                                    the final score. This suggests that simply extending time
Agent sustains deeper and more effective exploration than                                                   does not address the main failure mode when the agent does
OpenHands, with more loops (8.77 vs 3.69) and a higher im-                                                  not diagnose missing CoT supervision as the bottleneck.
provement rate (24.37% vs 17.31%) under the same toolset.
The modest extra iterative cost ($5.73 vs $3.92) reflects a                                                 4.4. Ablation Study
trade-off for tasks where shallow exploration stalls. These
                                                                                                            We conduct a comprehensive ablation study to disentangle
patterns suggest that FT-Agent is most valuable when data is
                                                                                                            the impact of training data size, planning backbone, and tar-
imperfect (e.g., AIME) or constraints are rigid (e.g., Chem-
                                                                                                            get model scale. We select five representative tasks covering
istry), while simpler baselines can remain competitive on
                                                                                                            diverse modalities: reasoning (AIME), retrieval (Patent-
cleaner standardized tasks.
                                                                                                            PI4PC), structural constraints (Chem-Mol Edit), knowledge
                                                                                                            (Financial QA), and code generation (Table-Vis). Results
4.3. Frontier Agents and Backbone Generality                                                                are summarized in Figure 3.
Reviewers raised whether FT-Agent’s gains reflect fine-                                                     1. Quality over Quantity (Data Scaling). Expanding the
tuning-oriented design or simply a stronger compared agent                                                  training set from 2k to 5k samples yields mixed results.
framework/backbone. We therefore add controlled compar-                                                     While performance improves on pattern-intensive tasks like
isons on the five representative tasks used in our ablation                                                 Patent Retrieval and Molecule Editing, it degrades on Vi-
study. All autonomous systems use the same FT-Dojo task                                                     sualization. Under a fixed 12-hour budget, larger datasets
setup, Docker environment, LlamaFactory/OpenCompass                                                         consume significantly more compute time for processing
wrappers, data repository, and 12-hour wall-clock budget.                                                   and training, squeezing the window for strategy iteration.
We also include the LLM-assisted Manual SFT result as an                                                    The 2k budget strikes a better balance between data cover-
assisted-human reference on the same five tasks.                                                            age and exploration depth.
FT-Agent with GPT-5.2 achieves the highest average among                                                    2. Necessity of Frontier Intelligence (Backbone Sensitiv-
all autonomous systems and the assisted-human reference.                                                    ity). We substitute the planning backbone in the iterative
Among GPT-5.2-based agents, it outperforms both Open-                                                       loop while keeping the underlying data-cleaning models
Hands and Codex, suggesting that the bottleneck is not only                                                 fixed. GPT-5.2 achieves the best average among FT-Agent
code generation quality but also data-quality diagnosis and                                                 variants, while GPT-4o improves Finance QA but drops on
multi-level feedback interpretation. The LLM-assisted Man-                                                  procedural tasks such as Visualization and PI4PC; open-
ual SFT row performs best on AIME but requires human                                                        source backbones remain competitive but show larger task-
data-synthesis decisions, so we use it to scope the con-

                                                                                                     7

## Page 8

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents
                                                 (a) Impact of Data Scaling                                     (b) Impact of Planning Backbone                                 (c) Impact of Target Model Scale
                         80   FT-Agent (2k)                                                80   GPT-5.2                                                     80    7B Base       3B Base
                              FT-Agent (5k)                                                     GPT-4o                                                            7B FT-Agent   3B FT-Agent
                                                                                                DeepSeek-V3.2
                         70                                                                70   Qwen3.5-397B                                                70



                         60                                                                60                                                               60



                         50                                                                50                                                               50




        Test Score (%)
                         40                                                                40                                                               40



                         30                                                                30                                                               30



                         20                                                                20                                                               20



                         10                                                                10                                                               10



                          0                                                                 0                                                                0
                               AIME           Patent        Chem        Finance    Table          AIME          Patent       Chem        Finance    Table           AIME        Patent         Chem       Finance   Table
                                              PI4PC        Mol_Edit        QA       Vis.                        PI4PC       Mol_Edit        QA       Vis.                       PI4PC         Mol_Edit       QA      Vis.




Figure 3. Ablation study results. (a) Data scaling: Performance comparison between 2k and 5k training samples. (b) Planning backbone:
FT-Agent variants with proprietary/open-source planners. (c) Target model scale: Performance across target sizes. Panel (b) shows
FT-Agent variants only; cross-agent baselines are in Table 4, with full ablations in Appendix Table 10.

Table 4. Five-task controlled comparison. Frontier agents, FT-Agent planning backbones, and an assisted-human reference on
representative tasks. Scores are percentages; Avg. is over displayed tasks.

                                                Method                        Setting                           AIME PI4PC Mol Edit Finance Vis.                                                Avg.
                                                Base Model                    –                                    0.00            34.00           22.20         65.10           4.00 25.06
                                                Manual SFT LLM synthesis                                          20.00            28.00           50.00         67.57          20.00 37.11
                                                OpenHands GPT-5.2                                                  0.00            44.00           40.00         64.73          24.00 34.55
                                                Codex       GPT-5.2                                                6.67            42.00           40.00         65.63          26.00 36.06
                                                Claude Code Sonnet-4.6-thinking                                    0.00            51.00           44.00         68.32          36.00 39.86
                                                FT-Agent                      Qwen3.5-397B-A17B 6.67                               24.00           34.00         67.16          28.00 31.97
                                                FT-Agent                      DeepSeek-V3.2      6.67                              45.00           50.00         65.47          12.00 35.83
                                                FT-Agent                      GPT-4o            13.33                              29.00           50.00         70.95          20.00 36.65
                                                FT-Agent                      GPT-5.2           13.33                              44.00           53.33         67.53          36.00 42.83


specific variance. These results indicate that the framework                                                                      capacity for iterative self-improvement through historical
transfers across planners, but high-level orchestration qual-                                                                     memory. Starting from a catastrophic failure (2% due to vo-
ity remains critical for navigating the optimization landscape                                                                    cabulary corruption), the agent diagnosed and addressed is-
effectively.                                                                                                                      sues across iterations: correcting data format misalignment,
                                                                                                                                  introducing RDKit validation to filter invalid SMILES,
3. Robustness Across Tested Model Scales. FT-Agent
                                                                                                                                  fixing insufficient training steps, and aligning operation
demonstrates strong generalizability, consistently unlocking
                                                                                                                                  distributions with the benchmark. This progression ulti-
significant gains on the smaller 3B model (e.g., +40% on
                                                                                                                                  mately drove accuracy to 56%, demonstrating how cumula-
Mol Edit, +16% on Visualization). These results suggest
                                                                                                                                  tive learning enables increasingly robust solutions.
that FT-Agent can transfer across the tested target model
scales: its ability to diagnose weaknesses and synthesize                                                                         The Causal Reasoning Gap. Conversely, PI4PC (Fig-
data remains effective on both 3B and 7B backbones under                                                                          ure 4b) illustrates the “shotgun debugging” failure mode.
our benchmark setting. Broader coverage over larger model                                                                         Facing stagnation, the agent indiscriminately applied
families and training regimes remains an important direction                                                                      generic techniques like NEFTune without causal diagno-
for future work.                                                                                                                  sis. These ungrounded interventions caused severe volatility
                                                                                                                                  (collapsing to 8%) and yielded zero net improvement. This
4.5. Case Study                                                                                                                   highlights a critical limitation: while current agents are
                                                                                                                                  efficient tool optimizers, they struggle to act as rigorous sci-
To understand the cognitive mechanisms governing these                                                                            entists when hypothesis testing requires deep causal insight.
outcomes, we analyze the step-by-step decision traces of
FT-Agent (detailed in Appendix D). This analysis reveals a
stark duality: the capacity for cumulative learning from                                                                          5. Related Work
historical experience versus the persistence of causal rea-                                                                       Interactive Environments for Agents. Recent benchmarks
soning deficits.                                                                                                                  have shifted from static dataset evaluation to interactive,
Cumulative Learning from Historical Experience. The                                                                               environment-based assessments. In software engineering,
trajectory on Mol Edit (Figure 4a) exemplifies the agent’s                                                                        SWE-Bench and SWE-agent demonstrated the potential of


                                                                                                                           8

## Page 9

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

                                   Mol_Edit: Cumulative Learning from Historical Experience                                                                         PI4PC: Pattern-Matching Failure
                          80                           max_steps               Prompt         Checkpoint                                            50                       0% net gain
                          70
                                                      +no-op filter          enrichment        +op ratio                                                                     after 7 loops




                                                                                                                          Validation Accuracy (%)
                                     Format fix




Validation Accuracy (%)
                                      +RDKit
                          60                                                                                                                        40
                          50                                                                                                                        30
                          40
                          30
                                                                                                           Base                                     20                               Collapse
                                                                                                                                                                                      (8%)
                                                                                                                                                         NEFTune
                          20        Vocab                                                                                                           10    (-11pp)
                                    corruption
                          10
                                                                                                                                                    0
                           0
                               0                  2             4            6            8          10                                                  0          1    2    3       4         5   6   7
                                                                      Loop                                                                                                        Loop
(a) Success (Mol Edit): The agent demonstrates cumulative learn-                                                         (b) Failure (PI4PC): The agent struggles with causal diagno-
ing through failure recovery and iterative refinement.                                                                   sis, yielding zero net gain.
Figure 4. Contrasting Learning Trajectories. (a) On Chemistry, the agent demonstrates cumulative learning, recovering from failure
and progressively refining its approach through domain tool integration and iterative optimization. (b) On Patent Classification, it exhibits
shotgun debugging, cycling through advanced techniques without identifying the root cause of overfitting.


agents to resolve real-world GitHub issues (Jimenez et al.,                                                           et al., 2024a;b) and EasyDataset (Miao et al., 2025) provide
2023; Yang et al., 2024). In the machine learning domain,                                                             composable operators for cleaning and formatting. Beyond
MLE-bench and MLE-Dojo evaluate agents on Kaggle-style                                                                processing, algorithmic approaches like DoReMi (Xie et al.,
competitions (Chan et al., 2024; Qiang et al., 2025). Recent                                                          2023) and LESS (Xia et al., 2024) introduce metrics to effi-
work has also begun to study how MLE agents should op-                                                                ciently select influential samples or optimize data mixtures.
timize, with Gome (Zhang et al., 2026) framing structured                                                             However, these methods operate in an open-loop manner us-
diagnostic reasoning as a gradient-like signal for directed                                                           ing static metrics, failing to address specific capability gaps
updates beyond scalar-score tree search. However, these                                                               exposed during training. FT-Agent instead treats data cura-
benchmarks typically operate under fixed-data assumptions,                                                            tion as an iterative optimization problem. It autonomously
confining agents to feature engineering or model search-                                                              formulates targeted strategies, such as synthesizing missing
ing. FT-Dojo distinguishes itself by formalizing the end-                                                             concepts or re-weighting hard examples, to resolve actual
to-end fine-tuning workflow, where agents must navigate                                                               failure modes based on evaluation feedback.
an open-ended decision space involving raw data curation,
processing logic, and training configuration.                                                                         6. Conclusion
Automated LLM Training Configuration. Implementing
                                                                                                                      In this work, we introduced FT-Dojo, an interactive bench-
fine-tuning pipelines from scratch presents a significant bar-
                                                                                                                      mark environment for evaluating autonomous agents on
rier for automation, requiring complex boilerplate code for
                                                                                                                      end-to-end LLM fine-tuning. Across 13 tasks in 5 domains,
distributed training, model loading, and gradient synchro-
                                                                                                                      FT-Dojo requires agents to select and construct training data
nization. To lower this barrier, integrated frameworks have
                                                                                                                      from raw sources, configure training pipelines, and refine
emerged (Zheng et al., 2024; Zhao et al., 2024), abstracting
                                                                                                                      their strategies from structured feedback, making data strat-
these complexities into unified, configuration-driven inter-
                                                                                                                      egy and training configuration joint optimization targets
faces (e.g., YAML or JSON). Building on these abstractions,
                                                                                                                      rather than fixed inputs.
recent works such as LaMDAgent (Yano et al., 2025) uti-
lize agents to optimize post-training workflows. However,                                                             We further proposed FT-Agent, a fine-tuning-oriented
these approaches typically rely on predefined templates or                                                            framework built around structured iteration planning, fail-
blind search, lacking the adaptability to dynamically tai-                                                            fast validation, and multi-level feedback analysis. Experi-
lor configurations to specific dataset characteristics (e.g.,                                                         ments show that FT-Agent provides a strong initial baseline,
sequence length) or hardware constraints (e.g., memory lim-                                                           achieving the best performance on 10 of 13 tasks and re-
its). FT-Agent bridges this gap by autonomously interfacing                                                           maining competitive in controlled comparisons with frontier
with training frameworks to derive optimal configurations                                                             agents and open-source planning backbones. Our case stud-
directly from environmental and data contexts.                                                                        ies further show both the promise and limits of autonomous
                                                                                                                      fine-tuning: agents can recover from failures through his-
Data-Centric LLM Fine-Tuning. Alongside training in-
                                                                                                                      torical experience, but still struggle with causal diagnosis
frastructure, data quality is recognized as a pivotal factor in
                                                                                                                      and long-horizon planning, leaving open-world data sourc-
model adaptation. Recent tooling has focused on standard-
                                                                                                                      ing and stronger reasoning mechanisms as important future
izing processing pipelines; for instance, Data-Juicer (Chen
                                                                                                                      directions.


                                                                                                                  9

## Page 10

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Impact Statement                                                      Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press,
                                                                        O., and Narasimhan, K. Swe-bench: Can language mod-
This paper introduces FT-Dojo, an interactive benchmark,                els resolve real-world github issues? arXiv preprint
and FT-Agent, an autonomous agent framework for end-                    arXiv:2310.06770, 2023.
to-end LLM fine-tuning. By automating the fine-tuning
pipeline—from data curation to model evaluation—our                   Jing, L., Huang, Z., Wang, X., Yao, W., Yu, W., Ma, K.,
work has the potential to lower the barrier for adapting                 Zhang, H., Du, X., and Yu, D. Dsbench: How far are
LLMs to specialized domains, broadening access to domain-                data science agents from becoming data science experts?
specific AI capabilities. However, reducing the human effort             arXiv preprint arXiv:2409.07703, 2024.
required for fine-tuning also lowers the cost of producing
                                                                      Li, H., Cao, H., Feng, B., Shao, Y., Tang, X., Yan, Z., Yuan,
models tailored for harmful purposes, such as generating
                                                                        L., Tian, Y., and Li, Y. Beyond chemical qa: Evalu-
misleading content in sensitive domains like finance or law.
                                                                        ating llm’s chemical reasoning with modular chemical
We believe that the transparency of our benchmark design
                                                                        operations. arXiv preprint arXiv:2505.21318, 2025.
and the emphasis on structured evaluation mitigate some
of these risks by encouraging reproducible and auditable              Lim, H., Nam, S., Na, S., Cho, J. Y., Yang, J. Y., Shin, H.,
fine-tuning practices. We encourage future work to incorpo-             Lee, Y., Kim, J., Lee, M., and Hong, H. Panorama:
rate safety-oriented evaluation criteria within similar bench-          A dataset and benchmarks capturing decision trails
marks.                                                                  and rationales in patent examination. arXiv preprint
                                                                        arXiv:2510.24774, 2025.
References                                                            Liu, A., Mei, A., Lin, B., Xue, B., Wang, B., Xu, B., Wu,
Art of Problem Solving Wiki. 2025 aime i, 2025. URL                     B., Zhang, B., Lin, C., Dong, C., et al. Deepseek-v3.
  https://artofproblemsolving.com/wiki/                                 2: Pushing the frontier of open large language models.
  index.php/2025_AIME_I. Accessed: 2026-01-26.                          arXiv preprint arXiv:2512.02556, 2025.

Chan, J. S., Chowdhury, N., Jaffe, O., Aung, J., Sherburn,            Miao, Z., Sun, Q., Wang, J., Gong, Y., Zheng, Y., Li, S.,
  D., Mays, E., Starace, G., Liu, K., Maksin, L., Patward-             and Zhang, R. Easy dataset: A unified and extensible
  han, T., et al. Mle-bench: Evaluating machine learning               framework for synthesizing llm fine-tuning data from
  agents on machine learning engineering. arXiv preprint               unstructured documents. In Proceedings of the 2025
  arXiv:2410.07095, 2024.                                              Conference on Empirical Methods in Natural Language
                                                                       Processing: System Demonstrations, pp. 960–968, 2025.
Chen, D., Huang, Y., Ma, Z., Chen, H., Pan, X., Ge, C., Gao,
                                                                      Qiang, R., Zhuang, Y., Li, Y., Zhang, R., Li, C., Wong,
  D., Xie, Y., Liu, Z., Gao, J., et al. Data-juicer: A one-
                                                                        I. S.-H., Yang, S., Liang, P., Zhang, C., Dai, B., et al.
  stop data processing system for large language models.
                                                                        Mle-dojo: Interactive environments for empowering llm
  In Companion of the 2024 International Conference on
                                                                        agents in machine learning engineering. arXiv preprint
  Management of Data, pp. 120–134, 2024a.
                                                                        arXiv:2505.07782, 2025.
Chen, D., Huang, Y., Pan, X., Jiang, N., Wang, H., Zhang,             Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng,
  Y., Ge, C., Chen, Y., Zhang, W., Ma, Z., et al. Data-juicer          B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H.,
  2.0: Cloud-scale adaptive data processing for and with               Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J.,
  foundation models. arXiv preprint arXiv:2501.14755,                  Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L.,
  2024b.                                                               Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R.,
                                                                       Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su,
Contributors, O. Opencompass: A universal evaluation
                                                                       Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and
  platform for foundation models. https://github.
                                                                       Qiu, Z. Qwen2.5 technical report, 2025. URL https:
  com/open-compass/opencompass, 2023.
                                                                       //arxiv.org/abs/2412.15115.
Duxiaoman-DI. Financeiq: Financial question answering                 Team, K., Bai, Y., Bao, Y., Chen, G., Chen, J., Chen,
  dataset. https://huggingface.co/datasets/                             N., Chen, R., Chen, Y., Chen, Y., Chen, Y., et al.
  Duxiaoman-DI/FinanceIQ, 2023.             Accessed:                   Kimi k2: Open agentic intelligence. arXiv preprint
  2026-01-26; Hugging Face Dataset.                                     arXiv:2507.20534, 2025.
Huang, J., Xiao, M., Li, D., Jiang, Z., Yang, Y., Zhang, Y.,          Wang, X., Li, B., Song, Y., Xu, F. F., Tang, X., Zhuge, M.,
  Qian, L., Wang, Y., Peng, X., Ren, Y., et al. Open-finllms:          Pan, J., Song, Y., Li, B., Singh, J., et al. Openhands: An
  Open multimodal large language models for financial                  open platform for ai software developers as generalist
  applications. arXiv preprint arXiv:2408.11878, 2024.                 agents. arXiv preprint arXiv:2407.16741, 2024.

                                                                 10

## Page 11

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Wu, X., Yang, J., Chai, L., Zhang, G., Liu, J., Du, X., Liang,
 D., Shu, D., Cheng, X., Sun, T., et al. Tablebench: A
 comprehensive and complex benchmark for table ques-
 tion answering. In Proceedings of the AAAI Conference
 on Artificial Intelligence, volume 39, pp. 25497–25506,
 2025.
Xia, M., Malladi, S., Gururangan, S., Arora, S., and Chen,
  D. Less: Selecting influential data for targeted instruction
  tuning. arXiv preprint arXiv:2402.04333, 2024.
Xie, S. M., Pham, H., Dong, X., Du, N., Liu, H., Lu, Y.,
  Liang, P. S., Le, Q. V., Ma, T., and Yu, A. W. Doremi:
  Optimizing data mixtures speeds up language model pre-
  training. Advances in Neural Information Processing
  Systems, 36:69798–69818, 2023.
Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao,
  S., Narasimhan, K., and Press, O. Swe-agent: Agent-
  computer interfaces enable automated software engineer-
  ing, 2024. URL https://arxiv.org/abs/2405.
  15793.
Yano, T., Ishibashi, Y., and Oyamada, M. Lamda-
  gent: An autonomous framework for post-training
  pipeline optimization via llm agents. arXiv preprint
  arXiv:2505.21963, 2025.
Zhang, H., Chen, J., Jiang, F., Yu, F., Chen, Z., Chen, G., Li,
  J., Wu, X., Zhiyi, Z., Xiao, Q., et al. Huatuogpt, towards
  taming language model to be a doctor. In Findings of the
  association for computational linguistics: EMNLP 2023,
  pp. 10859–10885, 2023.

Zhang, Y., Yang, X., Yang, X., Xian, B., Li, Q., Fang, S., Li,
  J., Wang, J., Xu, M., Liu, W., et al. Reasoning as gradient:
  Scaling mle agents beyond tree search. arXiv preprint
  arXiv:2603.01692, 2026.

Zhao, Y., Huang, J., Hu, J., Wang, X., Mao, Y., Zhang,
  D., Jiang, Z., Wu, Z., Ai, B., Wang, A., Zhou, W., and
  Chen, Y. Swift:a scalable lightweight infrastructure for
  fine-tuning, 2024. URL https://arxiv.org/abs/
  2408.05517.
Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z.,
  and Ma, Y. Llamafactory: Unified efficient fine-tuning of
  100+ language models. arXiv preprint arXiv:2403.13372,
  2024.




                                                                  11

## Page 12

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Appendix
A. Data Sources and Benchmark Details of FT-Dojo
This section elaborates on the construction details of the FT-Dojo suite. We provide a comprehensive overview of the
evaluation benchmarks, the corresponding training resources, our rigorous data processing pipeline, and detailed statistics
for each subtask.

A.1. Evaluation Benchmarks
A.1.1. AIME 2025 (M ATHEMATICS )
Source: American Invitational Mathematics Examination 2025 (Art of Problem Solving Wiki, 2025).
AIME 2025 serves as our primary testbed for high-difficulty mathematical reasoning. It contains competition-level problems
requiring multi-step derivation across algebraic manipulation, number theory, combinatorics, and geometry. Unlike standard
benchmarks, problems demand error-sensitive symbolic manipulation under strict exact-match evaluations, where answers
are integers between 0–999. Metrics: Accuracy, using a cascade evaluator that first applies symbolic verification and then
sends unresolved cases to a fixed LLM judge with the rubric in Appendix A.4.

A.1.2. PANORAMA (PATENT E XAMINATION )
Source: LG AI Research & KAIST (Lim et al., 2025).
PANORAMA simulates the professional patent examination process. It is constructed from U.S. patent records preserving
complete “decision trails” (applications, cited prior art, office actions). We evaluate on three subtasks:

  • PAR4PC (Prior Art Retrieval): Given a claim and 8 candidate patents, identify the specific prior art cited for rejection.
    Metrics: Accuracy (Exact Set Match), Macro-F1.

  • NOC4PC (Novelty/Obviousness Classification): Classify claims as ALLOW, §102 Rejection (Lacks Novelty), or
    §103 Rejection (Obvious). Metrics: Accuracy, Macro-F1.

  • PI4PC (Paragraph Identification): Identify the precise paragraph within a prior art document that supports the
    rejection. Metrics: Accuracy.

A.1.3. C HEM C OTB ENCH (C HEMISTRY )
Source: ChemCoTBench (Li et al., 2025).
Constructed from a massive corpus of 2 million molecular samples, ChemCoTBench represents a shift from traditional
knowledge retrieval to generative chemical reasoning. Unlike simple QA tasks, this benchmark evaluates an LLM’s
ability to perform precise, multi-step molecular manipulations under strict physical constraints. It introduces a Modular
Chain-of-Thought (CoT) framework, where complex tasks (e.g., optimizing drug solubility while maintaining scaffold
integrity) are decomposed into discrete, verifiable operations. Crucially, the benchmark enables intermediate consistency
checking, allowing automatic verification of chemical validity (e.g., valence rules, ring integrity) at each reasoning step,
rather than relying solely on the final answer. The suite covers the full lifecycle of chemical discovery:

  • Mol Und (Molecular Understanding): Tasks include functional group counting, ring counting, and SMILES
    equivalence. Metrics: MAE (for counting), Tanimoto Similarity, Accuracy.

  • Mol Edit (Molecular Editing): Structural modification (add/delete/substitute) while maintaining validity. Metrics:
    Accuracy, Tanimoto Similarity.

  • Mol Opt (Molecular Optimization): optimizing molecules for properties (e.g., QED, LogP, DRD2) while preserving
    scaffolds. Metrics: Success Rate, Valid SMILES %, Scaffold Match.

  • Reaction: Forward/retrosynthesis and condition recommendation. Metrics: Fingerprint Tanimoto Similarity, BLEU,
    Accuracy.

                                                            12

## Page 13

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

A.1.4. F INANCE IQ (F INANCE )
Source: Chinese financial certification examination dataset (Duxiaoman-DI, 2023).
Covers 10 Chinese financial certification exams (CPA, banking, securities, fund, insurance, economic analyst, taxation,
futures, financial planner, actuarial). Tests precision under numeric pressure and multi-step financial reasoning with
finance-specific assumptions. Task: Multiple-choice questions. Metrics: Accuracy.

A.1.5. TABLE B ENCH (TABLE QA)
Source: TableBench (Wu et al., 2025).
A comprehensive benchmark for structure-aware grounding over tabular evidence, consisting of 886 test cases across 18
fields.


  • DA (Data Analysis): Trend forecasting and statistical analysis (±10% tolerance). Metrics: Accuracy.

  • FC (Fact Checking): Verifying statements against tabular data. Metrics: Exact Match.

  • NR (Numerical Reasoning): Arithmetic operations (sum, average, percentage). Metrics: Accuracy.

  • Vis (Visualization): Generating Python code to plot charts. Metrics: Pass@1 (Executable Code Rate).


A.2. FT-Dojo Raw Training Resources
To facilitate reproducible research and model adaptation, FT-Dojo incorporates a curated repository of raw training data
spanning all five target domains. We aggregated these resources from diverse high-quality public sources to serve as the
foundational curriculum (or seed supervision) for agent fine-tuning. Prior to inclusion, all raw data underwent a rigorous
decontamination process to strictly eliminate any overlap with the FT-Dojo evaluation benchmarks, ensuring the integrity of
our zero-shot assessment.

A.2.1. M ATHEMATICS : D EEP S CALE R
Source: agentica-org/DeepScaleR-Preview-Dataset.
DeepScaleR contains approximately 40,000 competition-level mathematics problems compiled from AIME (1984–2023),
AMC, Omni-MATH, and Still datasets. Problems cover algebraic manipulation, number theory, combinatorics, and
geometric reasoning.
Data Quality: 82% of samples lack solutions; remaining 18% contain brief summary-style answers. CoT generation is
required for effective training.

A.2.2. PATENT E XAMINATION : PANORAMA
Source: LG-AI-Research/PANORAMA (Lim et al., 2025).
We use the official training split containing 254,449 samples across three tasks: Prior Art Retrieval (54,028), Novelty
Classification (136,211), and Paragraph Identification (64,210).
Data Quality: Q&A pairs only, no chain-of-thought annotations. We remove validation and test splits to prevent data
leakage.

A.2.3. C HEMISTRY: C HEM C OTDATASET
Source: OpenMol/ChemCoTDataset (Li et al., 2025).
The dataset provides approximately 23,000 samples with structured chain-of-thought annotations, distilled from Gemini-2.5-
pro, DeepSeek-R1, and Claude-3.7-sonnet, validated by 13 chemistry PhD candidates with ¿90% accuracy.
Data Quality: Medium-high quality CoT annotations covering molecular understanding, editing, optimization, and reaction
prediction.

                                                            13

## Page 14

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

A.2.4. F INANCE : F INANCE IQ
Source: Duxiaoman-DI/FinanceIQ (Duxiaoman-DI, 2023).
Contains 6,179 multiple-choice questions from 10 Chinese financial professional certification exams, including CPA,
banking, securities, insurance, and actuarial domains.
Data Quality: Q&A pairs only, without reasoning chains. CoT generation recommended for optimal performance.

A.2.5. TABLE QA: TABLE I NSTRUCT
Source: Multilingual-Multimodal-NLP/TableInstruct (Wu et al., 2025).
Large-scale instruction tuning dataset for table question answering, containing diverse examples across fact checking,
numerical reasoning, data analysis, and visualization tasks.
Data Quality: Includes reasoning chains with multiple instruction types: Direct Prompting (DP), Textual CoT (TCoT),
Program-of-Thought (PoT), and Symbolic CoT (SCoT).

A.3. Data Processing
A.3.1. E VALUATION DATA C ONSTRUCTION
To construct robust evaluation splits, we first apply context-length filtering to ensure compatibility across model architectures.
We then randomly partition the filtered instances into non-overlapping validation and test sets. Specifically, for each
benchmark (or subtask), we allocate samples according to:

                                              Nval = Ntest = min(100, ⌊n/2⌋)                                                  (3)

where n is the total number of filtered instances. The validation set is the only held-out feedback visible during optimization;
the test set remains hidden until final assessment. This strict separation ensures that agents are evaluated on held-out data
distinct from their training resources.

A.3.2. T RAINING DATA P REPARATION
All training datasets are processed with the following protocol:

  1. Leakage Prevention: For datasets with official splits (e.g., PANORAMA), validation and test partitions are explicitly
     removed.

  2. Format Normalization: All instances are converted to a unified instruction-response format.

  3. Quality Filtering: Samples exceeding context windows or containing corrupted text are filtered out.

A.4. Evaluator Rubrics and LLM-as-a-Judge Prompt
Most FT-Dojo tasks use deterministic or task-specific evaluators, including exact match, accuracy, Macro-F1, executable-code
checks, molecular validity, Tanimoto similarity, BLEU, and domain-specific chemistry metrics. The main LLM-as-a-judge
case is AIME 2025. For AIME, we use a cascade evaluator: a rule-based mathematical verifier is applied first, and only
unresolved cases are sent to a fixed GPT-5 judge. The judge prompt, rubric, and model are identical across all methods.
   AIME Judge Prompt Template

   You are grading a math answer for correctness.
   Input: original problem, gold answer, and model prediction.
   Rubric: Mark the prediction as correct if it is mathematically equivalent to the gold answer. Ignore formatting
   differences such as boxed notation, whitespace, punctuation, or explanatory text. For multi-part answers, all required
   parts must be correct. Do not give partial credit. If the answer is ambiguous, missing, or not mathematically equivalent,
   mark it incorrect.


                                                               14

## Page 15

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents


   Output: Return exactly one label: CORRECT or INCORRECT.


A.5. Dataset Statistics
We provide a granular breakdown of the evaluation datasets. Table 5 summarizes the sample counts for each specific subtask
within ChemCoTBench, TableBench, and PANORAMA, ensuring balanced coverage of all task types (e.g., distinguishing
between Forward Synthesis and Retrosynthesis).
In contrast, AIME 2025 (Mathematics) and FinanceIQ (Finance) are evaluated as holistic benchmarks. Due to their unified
nature, they are not stratified into subtasks but are maintained as monolithic validation and test sets to evaluate overall
domain proficiency. For AIME 2025, only 30 filtered instances remain after compatibility filtering, yielding 15 validation
and 15 test instances under the split rule above. FinanceIQ follows the 100/100 validation/test allocation.

B. Implementation Details of Baselines
To ensure a rigorous evaluation, we benchmark FT-Agent against primary baselines that represent manual fine-tuning, general-
purpose autonomous coding agents, and stronger frontier-agent systems. This section details the specific implementation
protocols for each.

B.1. Manual SFT (Constrained Manual Baseline)
The Manual SFT baseline is a constrained, resource-matched manual workflow for model adaptation. It was executed
by senior NLP researchers, each possessing over five years of experience in large language model training and data
engineering. We do not treat this baseline as a human-expert upper bound or as a fully assisted human best-effort result; it is
a documented comparison point under the same wall-clock budget and a restricted iteration protocol.

B.1.1. W ORKFLOW P ROTOCOL
Practitioners followed a standard three-stage pipeline, mirroring the agent’s logical steps but executed manually:

   • Data Engineering: Developing custom Python pipelines (using Pandas/Datasets) to implement robust rule-based
     cleaning strategies (e.g., removing noise, deduplication) prior to converting and tokenizing the data into the standard
     Alpaca schema.

   • Training Configuration: Manually configuring train config.yaml for LLaMA-Factory, selecting hyperparame-
     ters (e.g., Learning Rate 1e-4 ∼ 5e-5, cosine scheduler) based on best practices for 7B models.

   • Evaluation: Executing OpenCompass scripts to validate performance and iteratively refining the approach based on
     results.

B.1.2. C ONSTRAINTS
To match the agent’s compute budget, human practitioners were restricted to a “Two-Pass” protocol: one initial exploration
run and one refinement run. They were prohibited from performing open-ended grid searches and did not use LLM assistance
for data synthesis or cleaning in the main Manual SFT baseline. This restricted iteration count is an important limitation of
the comparison.

B.1.3. LLM-A SSISTED M ANUAL SFT C HECK
To assess the effect of LLM access for manual data synthesis and cleaning, we additionally ran an LLM-assisted Manual
SFT variant on five representative tasks. This variant improves substantially on tasks where missing CoT or difficult data
cleaning limits the original constrained protocol. FT-Agent remains competitive while operating fully autonomously, but we
use these results to scope the Manual SFT comparison rather than to claim superiority over unconstrained human experts.
The FT-Agent row in this table uses the representative single-run scores from the corresponding comparison; the main table
reports 3-run mean and standard deviation.

                                                              15

## Page 16

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents




Table 5. Detailed statistics of evaluation datasets across FT-Dojo benchmarks.

Task Category       Subtask                              Total Valid Test
1. ChemCoTBench (Chemistry)
                    Murcko scaffold                         40       20     20
                    equivalence                            100       50     50
Mol Und             fg count                               100       50     50
                    ring count                              20       10     10
                    ring system scaffold                    60       30     30
                    Subtotal                               320     160    160
                    add / delete                            40       20     20
Mol Edit
                    substitute                              60       30     30
                    Subtotal                               100       50     50
                    drd / gsk / jnk                        300     150    150
Mol Opt
                    logp / qed / solubility                300     150    150
                    Subtotal                               600     300    300
                    Forward / Retrosynthesis               200     100    100
                    Reaction Condition                      90      45     45
Reaction
                    Next Step Product                       85      42     42
                    Mechanism Selection                    100      50     50
                    Subtotal                               475     237    237
2. TableBench (Table QA)
                    Anomaly / Causal / Correlation         142       70     70
Data Analysis       Descriptive / Impact / Stat            151       75     75
                    TrendForecasting                        50       25     25
                    Subtotal                               343     170    170
Fact Checking       MatchBased / Multi-hop                  96       48     48
               Aggregation / Arith / Comp                  150       75     75
Num. Reasoning Count / Domain / Multi-hop                  150       74     74
               Ranking / Time-based                         97       48     48
                    Subtotal                               397     197    197
Visualization       ChartGeneration                         50       25     25
3. PANORAMA (Patent)
                    Prior Art Retrieval (PAR4PC)         2896      100    100
Patent Exam.        Novelty Classif. (NOC4PC)            2884      100    100
                    Paragraph ID (PI4PC)                 3402      100    100
                    Total                                9182      300    300




                                     16

## Page 17

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Table 6. Manual SFT with LLM-assisted synthesis on five representative tasks. All scores are percentages and higher is better. FT-Agent
values are representative single-run scores for this focused comparison.

                   Method                              AIME      PI4PC      Table Vis.   Finance QA      Mol Edit
                   Manual SFT (original)                 0.00     28.00        8.00          71.68          0.00
                   Manual SFT (w/ LLM synthesis)        20.00     28.00       20.00          67.57          50.00
                   FT-Agent                             13.33     44.00       36.00          67.53          53.33



B.2. OpenHands Baseline (Tool-Augmented Harness)
We evaluated OpenHands (v0.14) using the CodeAct architecture. To ensure a fair comparison, we implemented a
specialized evaluation harness that standardizes the execution environment.

B.2.1. E XPERIMENTAL S ETUP
We evaluate OpenHands on the FT-Dojo benchmark under conditions strictly identical to FT-Agent. To isolate the impact of
the framework design, we enforce a rigorous control protocol where both agents operate on a single NVIDIA B200 GPU
with a strict 12-hour wall-clock limit and use the same base model (Qwen2.5-7B-Instruct) for fine-tuning.
Furthermore, we align all external resources:

   • Reasoning Engine: Both agents invoke identical LLM API endpoints (GPT-5.2) for reasoning.

   • Data Access: Both access the exact same Data Repository D (raw training files) without prior processing.

   • Environment: Both operate within the same Dockerized sandbox with standard Python/Bash utilities.

OpenHands is prompted with the same high-level task descriptions and has access to the same fine-tuning/evaluation
wrappers, but it operates without FT-Agent’s structured iteration planning, fail-fast validation, best-configuration tracking,
and structured feedback analysis.

B.2.2. T OOL -AUGMENTED E NVIRONMENT
To prevent the baseline from failing due to trivial CLI syntax errors, we injected two high-level custom tools into the
sandbox:

   • LlamaFactoryTool: A Python wrapper around llamafactory-cli. It accepts structured parameters (e.g.,
     finetuning type=’lora’) via a Pydantic interface, automatically generating valid train config.yaml
     and parsing training logs.

   • OpenCompassTool: A wrapper for the evaluation stack using Jinja2 templates to dynamically generate configurations
     and parse CSV summaries into structured JSON observations.

B.2.3. S CAFFOLDED W ORKFLOW O RCHESTRATION
Unlike FT-Agent which autonomously plans its research trajectory, we design a systematic scaffold in main.py to guide
the OpenHands baseline through standard research stages:

  1. Data Cleaning Stage: The harness instantiates a DataAgent with a fixed prompt requesting a cleaning script. It
     includes a “Self-Correction Loop” that feeds stderr back to the agent upon failure.

  2. Training Stage: The harness transitions to a TrainAgent equipped with LlamaFactoryTool, providing heuristic
     parameter guidelines (e.g., based on sample size).

  3. Evaluation Stage: The harness invokes an EvalAgent to run OpenCompassTool.

                                                                  17

## Page 18

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

B.2.4. R ESILIENCE M ECHANISMS
To prevent failure from transient network issues, the harness includes enterprise-grade resilience features:

   • Load Balancing: API requests are distributed across multiple local endpoints (ports 4000–4007).

   • Transient Retry: A run conversation with retry wrapper automatically handles LLM API timeouts with
     linear backoff.

B.3. Frontier-Agent and Open-Source Backbone Baselines
For the additional comparison in Table 4, we evaluate Codex (GPT-5.2) and Claude Code (Claude Sonnet-4.6-thinking)
as frontier-agent baselines. We also evaluate FT-Agent with two open-source planning backbones, DeepSeek-V3.2 and
Qwen3.5-397B-A17B. All systems are run on the same five representative tasks used in the ablation study: AIME, PI4PC,
Mol Edit, Finance QA, and Table Visualization.
All runs use identical external conditions: the same FT-Dojo data repository, task specifications, Docker sandbox, LlamaFac-
tory/OpenCompass wrappers, target base model, validation/test splits, and 12-hour wall-clock budget. Frontier coding agents
are given the same high-level task objective and access to the same training/evaluation commands available to OpenHands.
FT-Agent variants differ only in the planning backbone; the fail-fast validation, feedback aggregation, and fine-tuning
wrappers are held fixed.

B.4. Extended-Budget Check
To examine whether the fixed 12-hour budget inherently favors fail-fast validation, we doubled OpenHands’ budget to
24 hours on AIME 2025 for two independent runs. The additional time did not yield a meaningful improvement: the
best checkpoint in both runs appeared within the first 6 hours, and later iterations did not improve the final score. This
supports the interpretation that OpenHands’ main bottleneck on AIME is not raw wall-clock time but failure to diagnose the
missing-CoT supervision issue in the raw training repository.

C. Detailed Comparison: OpenHands vs. FT-Agent
This section provides a granular analysis of why general-purpose coding agents, exemplified by OpenHands (Wang et al.,
2024), struggle with the specialized domain of autonomous LLM fine-tuning. Crucially, this performance gap persists even
when OpenHands is equipped with the identical toolset (LlamaFactory/OpenCompass wrappers) as FT-Agent, isolating the
root cause to cognitive architecture rather than tool availability.

C.1. Quantitative Results Comparison
To substantiate the cognitive disparity between the two agents, Table 7 presents the comprehensive evaluation results across
all 13 tasks. Unlike the condensed view in the main text, this table reports performance on both Validation and Test splits,
alongside auxiliary metrics (e.g., Macro F1, Tanimoto Similarity) to provide a holistic assessment of agent capability.

Robustness and Generalization Crucially, the performance gains of FT-Agent are consistent across validation and test
sets. For instance, in the Patent Examination domain, FT-Agent maintains a clear lead in both Exact Match and F1 scores
across splits, mitigating potential concerns of test-set overfitting. Similarly, in the Math domain (AIME), FT-Agent achieves
a non-trivial success rate (13.30% Test) where the baseline fails completely (0.00%), indicating that the performance gap
stems from a fundamental difference in problem-solving strategy rather than stochastic variance.

Metric-Level Granularity      The inclusion of granular metrics reveals specific behavioral differences:

   • Structure vs. Sequence: In Chemistry, FT-Agent outperforms the baseline in tasks requiring structural understanding
     (Mol Und and Mol Opt), evidenced by lower error rates (MAE) and higher validity scores. This aligns with the
     hypothesis that our agent effectively leverages tools to verify chemical validity.

   • Trade-offs: Conversely, OpenHands retains a slight edge in the Reaction prediction task (higher BLEU). This suggests

                                                             18

## Page 19

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Table 7. Full performance comparison on Validation and Test sets. This table supplements the main text by including all auxiliary
metrics (e.g., Macro F1, Tanimoto, BLEU) and validation splits to demonstrate robustness. Bold indicates the best performance in each
column (for error metrics like MAE, lower is better).

                                                                         OpenHands          FT-Agent
                            Domain          Task         Metric
                                                                          Val     Test     Val      Test
                            Math            AIME         Acc             0.00     0.00    20.00    13.30
                                                         Exact Match     33.00   34.00    54.00    50.00
                                            PAR4PC
                                                         Macro F1        50.93   50.51    75.43    70.34
                            Patent Exam.                 Acc             35.35   34.34    40.00    44.00
                                            NOC4PC
                                                         Macro F1        27.47   28.44    37.18    41.09
                                            PI4PC        Acc             47.00   44.00    40.00    44.00
                                            DA           Acc             33.78   23.22    33.66    34.48
                                            FC           Acc             79.17   66.67    85.42    83.33
                            Table QA        NR           Acc             45.00   40.00    52.00    46.00
                                                         Pass@1          8.00    24.00    20.00    36.00
                                            Vis
                                                         ECR@1           56.00   60.00    48.00    60.00
                            Finance         QA           Acc             74.40   64.73    75.00    67.53
                                                         MAE FG↓         0.18    0.10     0.26     0.04
                                                         MAE Ring↓       0.90    1.20     0.60     0.80
                                            Mol Und      Tanimoto         0.08    0.10     0.35     0.42
                                                         Acc Equiv       48.00   56.00    56.00    50.00
                                                         Acc Scaffold    70.00   80.00    83.33    73.33
                            Chemistry                    Acc             55.56   40.00    62.22    53.33
                                            Mol Edit
                                                         Tanimoto        52.63   53.29    50.11    57.19
                                                         SR              34.67   31.00    38.33    36.67
                                            Mol Opt      Valid Rate      81.67   80.67    89.67    91.67
                                                         Scaffold        25.67   20.67    32.00    33.00
                                                         FTS             29.97   35.59    16.89    18.11
                                            Reaction     BLEU            33.16   44.93    33.79    33.10
                                                         Acc             64.00   60.00    36.00    38.00


     that while FT-Agent’s reasoning architecture excels at structured, rule-bound scientific tasks, the baseline’s direct
     approach remains competitive for pure sequence-to-sequence text generation tasks.

C.2. Efficiency Analysis: The Cost of Trial-and-Error
Efficiency in autonomous LLM fine-tuning is not merely about inference speed, but about the success rate of experimental
trials. Table 8 compares the number of effective iterations (defined as training runs that complete the full pipeline without
fatal runtime errors) within a fixed 12-hour budget.
FT-Agent achieves approximately 2.2× more effective iterations (6.0 vs. 2.75) than OpenHands. This disparity stems from
the high ”opportunity cost” of failure in OpenHands:

   • Fail-Slow vs. Fail-Fast: OpenHands often generates syntactically correct but semantically flawed training scripts
     (e.g., tensor shape mismatches or OOM errors) that crash only after hours of execution. FT-Agent utilizes Progressive
     Validation to verify data integrity and configuration validity before allocating GPU resources, ensuring that most
     execution time is spent on valid training runs.
   • Exploration Efficiency: In complex tasks like Table QA FC, where data formatting is brittle, OpenHands wasted
     significant time debugging basic file parsing errors (yielding only 2 effective runs). In contrast, FT-Agent’s specialized
     validation and orchestration pipeline resolved these formatting issues pre-training, allowing for 7 complete experimental
     loops.

                                                                  19

## Page 20

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Table 8. Number of effective iterations (completing full training) within the 12-hour budget. FT-Agent achieves 2.2× more effective
iterations due to fail-fast validation.

                                            Task            OpenHands       FT-Agent
                                            AIME                  2             5
                                            PAR4PC                3             6
                                            Table QA FC           2             7
                                            Mol Opt               4             6
                                            Average              2.75          6.0


C.3. Qualitative Analysis: The Cognitive Gap
The quantitative gap is a symptom of a fundamental difference in cognitive architecture. While both agents utilize the same
underlying toolset (LlamaFactory), their approach to the problem space differs radically:

C.3.1. O PEN H ANDS : T HE G ENERALIST T RAP
OpenHands treats fine-tuning as a generic software engineering task.

   • Blind Code Generation: It generates data processing scripts and training configurations as arbitrary code. Lacking
     domain knowledge, it frequently hallucinates non-existent hyperparameters or uses deprecated library arguments,
     leading to ”silent failures” where training runs but yields no learning.

   • Reactive Debugging: When errors occur, OpenHands attempts to fix the Python syntax rather than the underlying
     logic (e.g., fixing a JSON error instead of adjusting the learning rate), often entering infinite debugging loops.

C.3.2. FT-AGENT: D OMAIN -S PECIFIC R EASONING
FT-Agent is architected specifically for the lifecycle of model adaptation, addressing the unique challenges of autonomous
LLM fine-tuning:

   • Schema-Aware Configuration: Instead of guessing parameters, FT-Agent grounds its generation in the strict schema
     of the training framework, substantially reducing configuration errors.

   • Task-Aware Data Processing: It proactively analyzes the raw data distribution (e.g., sequence length, label balance) to
     tailor the preprocessing strategy, preventing data-induced crashes.

   • Cumulative Learning: Unlike the stateless retries of OpenHands, FT-Agent maintains a history of failed experiments.
     If a specific learning rate leads to divergence, it updates its internal belief state to avoid that region in subsequent
     iterations.

This analysis validates the design principles introduced in §3.2: effective autonomous LLM fine-tuning requires specialized
mechanisms, specifically verification, structural grounding, and historical reflection, that extend beyond the capabilities of
general-purpose code generation.

D. Case Study: Emergent Capabilities and Limitations
Beyond the focused optimization of individual benchmarks, we observe distinctive behavioral patterns that highlight both
the emergent intelligence and the cognitive boundaries of FT-Agent. This section analyzes its capability for autonomous
cross-domain transfer, its vulnerability to myopic planning, and provides a complete optimization trace on TableBench.

D.1. Emergent Capability: Autonomous Cross-Domain Transfer
A distinctive capability of FT-Agent is its ability to autonomously identify and leverage data sources beyond the target
task’s primary dataset—a strategy that addresses the open-ended data decision space. We observe this behavior in two
complementary forms.

                                                                20

## Page 21

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

                         TableBench Num. Reasoning                                                              ChemCOT Mol_Opt
                               (Cross-Domain)                                                                    (Cross-Sub-Task)
             TableInstruct (in-domain)                                                   mol_opt (primary)
             DeepScaleR (out-of-domain)    "Bottleneck: arithmetic,                      mol_edit (auxiliary)   "Optimization needs structure
                                           not table comprehension"                                             understanding + editing skills"
                                                                                         mol_und (auxiliary)

                                 TableInstruct                         200                                 mol_opt                                 mol_edit    100
                                  1,800 (90%)                         (10%)                              1,400 (70%)                              500 (25%)   (5%)




       Acc                      Base: 32.0%                                         SR                  Base SR: 23.3%
                                 +14pp                                                                  +13pp
                                          FT-Agent: 46.0%                                                         FT-Agent SR: 36.7%

         0              500               1000            1500          2000         0               500                1000               1500               2000
                               Samples / Scaled Accuracy                                                   Samples / Scaled Success Rate

Figure 5. Cross-domain data strategies autonomously discovered by FT-Agent. Left: for Numerical Reasoning, the agent mixes
TableInstruct (90%) with out-of-domain DeepScaleR (10%). Right: for Mol Opt, the agent combines three chemistry sub-tasks with
task-informed proportions.



On TableBench Numerical Reasoning (Figure 5, left), the agent diagnosed that the core bottleneck was arithmetic reliability
rather than table comprehension. In Loop 1, it autonomously mixed TableInstruct (1,800 samples, 90%) with DeepScaleR
(200 samples, 10%)—a competition-level mathematics dataset (AIME, AMC) entirely outside the table QA domain—and
applied a unified instruction prompt to bridge the format gap. This cross-domain augmentation yielded 46.00% accuracy, a
14-point gain over the 32.00% baseline.
On ChemCOT Mol Opt (Figure 5, right), the agent employed a cross-sub-task variant of the same principle. By Loop 4, it
recognized that molecular optimization requires understanding molecular structure (from Mol Und) and editing operations
(from Mol Edit), constructing a composite training set of mol opt (1,400 samples, 70%), mol edit (500 samples, 25%), and
mol und (100 samples, 5%). This yielded a success rate of 36.67% with 91.67% valid SMILES, compared to 23.33% SR
and 68.67% VS for the baseline.
Notably, this behavior is selective: for Reaction Prediction, Mol Edit, and Mol Und, the agent used only target-task data,
indicating that FT-Agent applies domain transfer only when it identifies a specific capability gap that external data can
address.

D.2. Visualization: Data Collapse from Myopic Filtering
On the Visualization task (Figure 6), Loop 8 generated a filtering pipeline where each step passed local sanity checks:
compile validation (reasonable), plot-call detection (reasonable), token limits (reasonable). Yet the agent failed to simulate
their joint effect, which reduced the dataset from ∼1,800 to just 27 samples, a 98.5% reduction causing Pass@1 to plummet
from 20% to 4%. This reveals that current LLMs optimize decisions locally without maintaining a coherent model of
cumulative state changes, a form of “planning myopia” that distinguishes them from deliberative reasoners capable of
multi-step lookahead.

D.3. Evidence for Pattern-Matching: PI4PC Decision Trace
To substantiate our claim that agents exhibit pattern-matching rather than causal reasoning, we present the complete decision
trajectory for PI4PC, including the agent’s hypotheses and the resulting performance changes.
Analysis. Table 9 reveals a consistent pattern: the agent describes what went wrong but does not explain why at a mechanistic
level. Several observations:
• Hedging language: The agent uses “likely,” “appears,” “can” rather than definitive causal claims, indicating it is guessing
  at correlations rather than verifying mechanisms.
• Symptom–remedy pattern: Each loop follows “observe symptom → apply generic ML technique” without asking why
  the symptom occurred for this specific task.
• Insights not operationalized: Loop 2 correctly identifies “train–eval objective mismatch,” yet subsequent loops (3–7)

                                                                               21

## Page 22

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents


                                                     Visualization: Data Collapse
                                      2500                         Data collapse                       35
                                                                                             Samples
                                                              27 samples, 4% Pass@1          Pass@1    30
                                      2000



                   Training Samples
                                                                                                       25


                                                                                                          Pass@1 (%)
                                      1500                                                             20
                                      1000                                                             15
                                                                                                       10
                                      500
                                                                                                       5
                                        0                                                              0
                                               1     2    3     4    5     6   7      8       10
                                                                      Loop
Figure 6. Visualization: training samples and Pass@1 over loops. Loop 8’s multi-stage filtering pipeline reduced training data from
∼1,800 to just 27 samples, causing Pass@1 to collapse from 20% to 4%.




  never design a targeted fix for this specific issue.
• Missing causal questions: The agent never asks: Why does PI4PC overfit with 2k samples? What links rationale fluency
  to paragraph selection? How does lexical overlap proxy actual task difficulty?
Contrast with Mol Edit. On Mol Edit, Loop 4’s introduction of RDKit validation succeeded precisely because it addressed
the causal bottleneck: invalid SMILES strings were causing silent evaluation failures. This domain-specific intervention
emerged not from explicit causal reasoning but from the agent’s exposure to chemistry-related training data that associated
molecular tasks with RDKit. The distinction illustrates that agents can occasionally discover effective interventions through
pattern-matching when the solution happens to be well-represented in their knowledge base, but struggle when task-specific
causal analysis is required.

D.4. TableBench: A Complete Optimization Trace
We present a detailed trace of FT-Agent optimizing the TableBench scenario. This subsection is structured chronologically
to demonstrate the agent’s progressive scientific discovery: starting from data selection, moving to context processing,
determining training architecture, and finally refining data composition to address orthogonal capability gaps.

D.4.1. P ROBLEM S ETUP AND TASK A BSTRACTION
TableBench evaluates Large Language Models on structure-aware reasoning, demanding precise operations under strict
output contracts. Upon initialization, the agent diagnoses a core mismatch: the seed dataset (TableInstruct) is
code-heavy (Program-of-Thought), whereas the benchmark requires natural language reasoning.

D.4.2. I NSIGHT I: S TRUCTURAL C OMPLEXITY D RIVES E FFECTIVE S AMPLING
Having identified the domain mismatch, the agent’s first step was to select the right training samples.
Observation. Initial runs using conventional semantic stratification (e.g., qtype) failed to improve performance. Failure
analysis revealed that errors were driven by execution failures (e.g., wrong row filtering) which correlated with the structural
complexity of the latent program, not the question topic.
Agent Strategy. The agent pivoted to a difficulty-driven sampling strategy.


                                                                      22

## Page 23

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Table 9. PI4PC: Complete decision trajectory over 8 loops. Each row shows the agent’s strategy (before execution) and feedback (after
execution). Despite diverse interventions, the agent returns to baseline performance. Note the hedging language in feedback (“likely,”
“appears”) indicating correlation-based rather than causal reasoning.

             Loop    Acc.    Strategy (Before)                        Feedback (After)
               0     34%     Gold-anchored rationale + LoRA           “accuracy improves from 29% baseline to 34%...eval
                                                                      loss improving early then degrading suggests over-
                                                                      fitting”
               1     23%     +NEFTune + longer examiner ratio-        “NEFTune noise likely pushed the model to learn
                             nales (“to reduce overfitting”)          fluent justifications rather than improve discrimina-
                                                                      tive paragraph choice”
               2     15%     Remove NEFTune, keep contrastive         “train–eval objective mismatch where model learns
                             rationales                               plausible narratives without reliably performing dis-
                                                                      criminative selection”
               3     8%      Strict quote-matching filter (“improve   “only 1111/2000 samples survived...set appears ex-
                             data quality”)                           tremely easy...which can train toward overlap short-
                                                                      cuts”
               4     30%     Hard-case sampling by lexical overlap    “shortfall consistent with supervision/distribution
                                                                      mismatch...sampled difficulty still concentrates in
                                                                      very-easy cases”
               5     32%     Quota-based difficulty sampler           “hard-case quota did not materialize (bucket ‘≥7’
                                                                      ended up 0)...shift likely didn’t target hardest
                                                                      regimes”
               6     16%     Selector-oriented evaluation (“align     “experiment did not actually execute the central hy-
                             train-eval”)                             pothesis...used eval loss instead of selector metric”
               7     34%     Revert to baseline configuration         “accuracy exactly matching prior SOTA (no improve-
                                                                      ment). Key hypothesis factor was not implemented”


   Agent Sampling Trace

   Insight: Semantic labels hide execution difficulty. Action: Sample based on computational complexity proxies: 1.
   Table Size: (Rows × Cols) to approximate context load. 2. Constraint Density: Count of conjunctions/comparatives.
   Goal: Target boundary cases that challenge retrieval while remaining solvable.


D.4.3. I NSIGHT II: P REPROCESSING M UST B E TASK -A DAPTIVE
With the samples selected, the agent next addressed how to fit them into the model’s context window without losing
information.
Observation. The agent realized that context management is not a universal preprocessing step but a task-dependent
variable. For Fact Checking, latent programs are short; for Numerical Reasoning, they involve long multi-step arithmetic
chains.
Agent Strategy. The agent differentiated its pipeline based on the subtask structure:

   • Fact Checking → Truncation: Remove redundant boilerplate to maximize batch size, as table structure is preserved.
   • Numerical Reasoning → Filtering: Truncation is destructive to long CoT traces. Samples exceeding the context
     window must be filtered out to ensure reasoning consistency.

D.4.4. I NSIGHT III: R IGID F ORMATTING D EMANDS F ULL PARAMETER U PDATES
Once data preparation was established, the agent had to determine the optimal learning dynamics for the training phase.
Observation. While PEFT (LoRA) is standard for small datasets, TableBench requires rigid adherence to complex output
formats (JSON/Exact Match).
Agent Strategy. The agent decided on Full Fine-Tuning, prioritizing ”muscle memory” over parameter efficiency.

                                                                 23

## Page 24

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

   Architectural Decision Trace

   Insight: Low-rank adapters struggle with rigid output contracts. Rationale: 1. Resource: The B200 GPU (178GB)
   allows full parameter updates. 2. Performance: Full tuning empirically yields better instruction adherence and
   formatting compliance. 3. Safety: Overfitting is mitigated via conservative learning rates rather than freezing
   parameters.


D.4.5. I NSIGHT IV: O RTHOGONAL C APABILITIES R EQUIRE DATA M IXING
Finally, in a recursive optimization loop, the agent recognized that single-source data could not cover all required
capabilities.
Observation. Numerical Reasoning poses a dual challenge: it requires both table grounding (finding the right numbers)
and arithmetic reliability (calculating correctly). The primary source (TableInstruct) provided strong grounding but
weak arithmetic reasoning.
Agent Strategy. To bridge this gap, the agent formulated a Mixed Dataset Strategy:
   Data Mixing Logic

  Insight: Capabilities are orthogonal. TableInstruct gives grounding; DeepScaleR gives calculation. Action:
  - Primary (80%): TableInstruct for table operations coverage. - Auxiliary (20%): DeepScaleR to inject
  arithmetic reliability. Result: A composite dataset that closes the capability gap without domain drift.


E. Full Ablation Results
Table 10 presents the complete numerical results of our ablation studies. It details the performance comparisons across three
key dimensions: training data scale (2k vs. 5k), agent reasoning backbone (GPT-5.2 vs. GPT-4o), and target model capacity
(7B vs. 3B), covering both validation and test splits for all five benchmarks.

Table 10. Ablation study and baseline comparison organized by model size (7B vs. 3B). Scores are reported as validation/test. The
Default setting represents FT-Agent with Qwen2.5-7B (2k samples, GPT-5.2 backbone). Scaling-5k and w/ GPT-4o are ablations on top
of the 7B Default. Bold indicates the best validation and best test performance per row independently.

                                                      7B Model Settings                                3B Model Settings
Domain      Task (Metric)
                                    Base         Default (2k)       Scaling-5k     w/ GPT-4o           Base             FT
Math        AIME25 (Acc)          0.00/0.00      20.00/13.33        6.67/13.33     6.67/13.33        0.00/0.00       6.67/0.00
Patent      PI4PC (Acc)          29.00/34.00     40.00/44.00        36.00/48.00    38.00/29.00     19.00/21.00      31.00/34.00
Finance     QA (Acc)             73.00/65.10     75.00/67.53        75.55/66.33    70.33/70.95     61.44/56.98      60.73/58.47
Chem        Mol Edit (Acc)       34.40/22.20     62.22/53.33        63.33/58.89    62.22/50.00     13.44/14.44      53.93/54.55
Table       Vis (Pass@1)          8.00/4.00      20.00/36.00        20.00/28.00    24.00/20.00       0.00/8.00      8.00/24.00



F. Complete Validation Set Results
While the main text reports performance on the held-out Test Set (Table 2), we provide the corresponding Validation Set
performance in Table 11.
Strict Blind-Test Protocol. To ensure rigorous evaluation, we enforced a strict separation of data splits during the agent’s
autonomous operation. The Validation Set was fully accessible to the agent, serving as the internal feedback mechanism for
diagnosing failure modes, selecting optimal checkpoints, and refining data strategies during iterative loops. In contrast, the
Test Set remained strictly invisible to the agent throughout the entire optimization process. All autonomous decisions—
including when to stop training and which model to deploy—were based solely on validation metrics, ensuring that the
reported test set performance reflects true generalization capability rather than overfitting to the evaluation metric.

                                                               24

## Page 25

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents

Table 11. Complete validation set results. We report validation performance across 13 tasks in 5 domains. FT-Agent results are
reported as mean±std over 3 independent runs. Bold indicates the best performance. ↑: higher is better; ↓: lower is better. Abbreviations:
DA/FC/NR = Data Analysis/Fact Checking/Num. Reasoning; P@1 = Pass@1; MAE = Mean Absolute Error (for FG/Ring Count); TMS
= Tanimoto Similarity; SR = Success Rate; VS = Valid SMILES; FTS = Fingerprint Similarity.
                       Math           Patent Examination                         Table QA                     Finance                                        Chemistry
                       AIME     PAR4PC     NOC4PC       PI4PC      DA          FC        NR          Vis.      QA                 Mol Und              Mol Edit       Mol Opt                Reaction
Method                 Acc↑       F1↑        F1↑        Acc↑       Acc↑       Acc↑       Acc↑       P@1↑       Acc↑      MAE↓     TMS↑        Acc↑      Acc↑        SR↑     VS↑           FTS↑      Acc↑
Qwen2.5-7B-Instruct     0        60.63      27.40      29.00      28.87      83.30      40.00       8.00      73.00      0.52      0.09      56.00      34.40      21.00      66.00      14.93       36.00
Manual SFT              0        68.00      27.11      22.00      34.14      72.92      45.00      20.00      73.14      0.44      0.00      46.00       2.23      12.67      31.67        0         10.00
OpenHands               0        50.93      27.47      47.00      33.78      79.17      45.00       8.00      74.40      0.54      0.08      59.00      55.56      34.67      81.67      29.97       64.00
FT-Agent            13.32±6.67 67.50±6.89 34.44±4.82 38.67±2.31 33.56±3.16 82.64±4.82 46.67±4.62 16.00±4.00 74.04±1.10 0.24±0.17 0.26±0.08 62.78±5.96 57.04±5.01 36.22±2.36 88.78±6.38 30.64±11.93 39.33±3.06




G. Prompt Architecture
This section presents the prompt architecture used by FT-Agent, organized according to the four-phase workflow described
in §3.2. Rather than relying on a single monolithic instruction, the framework implements a multi-stage prompting pipeline
that mirrors how human researchers conduct benchmark analysis, model development, and experimental validation.

G.1. Phase 1: Strategy Proposal
The strategy proposal phase generates unified hypotheses that jointly cover data processing and training configuration. To
avoid repeating mistakes, the agent maintains a memory of historical experiments, inherits the current best configuration as
baseline, and reviews sibling attempts before proposing changes.
   Unified Hypothesis Generation

   Generate a comprehensive hypothesis covering BOTH data processing AND training configuration.
   Data Processing: Prioritize quality over quantity; apply code-based sampling within budget; use strong models for
   CoT generation with outcome-based validation.
   Training Configuration: Select method based on sequence length and hardware constraints; balance overfitting risk
   against model quality based on dataset size.
   Output: Structured JSON with reason, hypothesis, and implementation task.


G.2. Phase 2: Implementation
The implementation phase translates hypotheses into executable code, comprising data processing scripts and training
configurations. Both support a debug mode for rapid validation before full execution.
   Data Processing

   Generate a Python script to process the dataset according to the hypothesis.
   Execution: Support debug mode (small subset) and full mode (within sample budget).
   Sampling: Apply quality-first or diversity-preserving strategies based on hypothesis.
   CoT Requirement: All training data must include step-by-step reasoning; validate that reasoning leads to correct
   answers.
   Output: Alpaca-format JSON (instruction/input/output fields).


   Training Configuration

   Generate a training configuration based on the hypothesis and hardware constraints.
   Method Selection: Choose based on required sequence length, dataset size, and hardware memory constraints.


                                                                                                    25

## Page 26

FT-Dojo: Towards Autonomous LLM Fine-Tuning with Language Agents


   Batch Size: Balance sequence length against batch size; use gradient accumulation for effective batch size.
  Validation: Create validation split; align evaluation and save strategies.


G.3. Phase 3: Fail-Fast Validation
The fail-fast validation phase implements progressive verification that catches errors early before full training runs. This
corresponds to Stage 2 in the main text (§3.2).
   Data Validation

  Validate data processing output through three levels: (1) Schema: Output file exists and follows expected format. (2)
  Content: Required fields present in each sample. (3) Quality: CoT reasoning exists and answers follow benchmark
  format.
   Hard failures (empty output, zero samples, missing CoT) require code fixes; soft warnings (high filter rate, skewed
   distribution) are logged but allow continuation.


   Config Validation

  Validate configuration through progressive stages: (1) Syntax: Configuration parses correctly. (2) Compatibility:
  Parameters valid for method and hardware. (3) Execution: Mini-batch training completes without error.
   On failure, provide error classification, root cause analysis, and suggested fixes.


G.4. Phase 4: Feedback Analysis
The feedback analysis phase evaluates completed experiments and generates actionable insights for the next iteration. This
corresponds to Stage 3 (Structured Feedback Aggregation) in the main text.
   Experiment Feedback

  Analyze experiment results and decide whether to accept as new SOTA.
   Decision: Accept if benchmark results exceed SOTA/baseline on primary metrics; reject otherwise.
   Multi-level Analysis: (1) Metrics: Compare overall performance against baseline/SOTA. (2) Error Patterns: Identify
   capability gaps from failed samples (avoid benchmark-specific suggestions). (3) Training Dynamics: Analyze loss
   curves for overfitting/underfitting signs.
   Output: Code summary, decision rationale with improvement suggestions, and accept/reject decision.




                                                              26
