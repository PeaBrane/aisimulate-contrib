## Description: <br>
Find, per GPU type, the smallest qualifying GPU count found in a tested AISim search space for serving an LLM at a given request rate within a latency SLA, simulated on CPU. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>

## Use Case: <br>
Engineers, solution architects and capacity planners use this skill to estimate how many GPUs of which type a model needs at a given request rate and latency SLA, and what that costs at prices they supply, before buying, leasing or allocating hardware. Version 0.1.0 covers request-rate (Poisson or constant) workloads only. <br>

### Deployment Geography for Use: <br>
Global <br>

### Requirements / Dependencies <br>
Requires API Key or External Credential: Optional <br>
Credential Type(s): Hugging Face token only for gated models whose configs are not bundled with AISim <br>
Network: package download when installing aisim; HTTPS to huggingface.co for models whose configs are not bundled <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: A simulated GPU count can be lower than what real hardware needs, which could lead to under-provisioning. <br>
Mitigation: The skill reports the count as the smallest qualifying configuration found in the tested search space, checks an explicit compliance share and SLA-compliant throughput agreed with the user at each tested load, states that SLA checks use aggregate means, and recommends measuring the final choice on hardware at the same request rate. <br>

Risk: Each search step is sampled, and AISim 0.14.0 can overshoot in a single large-budget min_gpus search. <br>
Mitigation: The bundled ladder raises the budget step by step and refines in ascending order; the skill never calls the result a proven minimum. <br>

Risk: Cost figures depend on prices the user supplies. <br>
Mitigation: The skill never invents prices and shows the formula it used. <br>

## Reference(s): <br>
- [AISim repository](https://github.com/ai-dynamo/aisim) <br>
- [AISim end-to-end accuracy overview](https://ai-dynamo.org/aisim/e2e-accuracy/) <br>
- [AISim FPE support matrix](https://ai-dynamo.org/aisim/fpe-support-matrix/) <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, YAML configuration, Analysis] <br>
**Output Format:** [Markdown table with inline bash and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Writes AISim output directories where the agent is told to; installing aisim also writes the Python environment and package cache] <br>

## Evaluation Metrics Used: <br>
[Filled by NVSkills CI on signing.] <br>

## Skill Version(s): <br>
0.1.0 (draft; targets aisim 0.14.0; full evals pending) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
