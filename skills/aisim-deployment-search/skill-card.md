## Description: <br>
Rank LLM serving configurations for one model on a GPU type and budget already chosen, simulated on CPU with NVIDIA AISim, with an optional approved GPU spot-check. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>

## Use Case: <br>
Engineers who serve LLMs with NVIDIA Dynamo, vLLM or SGLang use this skill to narrow a deployment search (backend, aggregated or disaggregated serving, parallelism, replicas, scheduler limits) on CPU before spending GPU time, then confirm a short list on real hardware. <br>

### Deployment Geography for Use: <br>
Global <br>

### Requirements / Dependencies <br>
Requires API Key or External Credential: Optional <br>
Credential Type(s): Hugging Face token for gated model configs; cluster credentials only for the optional GPU spot-check <br>
Network: package download when installing aisim; HTTPS to huggingface.co for models whose configs are not bundled; cluster API only for the optional spot-check <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Simulated rankings can differ from measured results, which could lead a user to pick a weaker configuration. <br>
Mitigation: The skill labels every result as simulated, reports the timing data provenance, and recommends a GPU spot-check of the short list before production or capacity decisions. <br>

Risk: The optional spot-check launches GPU workloads on a user's host or cluster. <br>
Mitigation: The skill stops for explicit approval of picks, GPU count and duration before any launch, never applies simulation YAML as a Kubernetes manifest, and cleans up after the benchmark. <br>

## Reference(s): <br>
- [AISim repository](https://github.com/ai-dynamo/aisim) <br>
- [AISim end-to-end accuracy overview](https://ai-dynamo.org/aisim/e2e-accuracy/) <br>
- [AISim FPE support matrix](https://ai-dynamo.org/aisim/fpe-support-matrix/) <br>
- [AIConfigurator paper](https://arxiv.org/abs/2601.06288) (AIConfigurator is AISim's former name) <br>
- [AIPerf](https://github.com/ai-dynamo/aiperf) <br>

## Skill Output: <br>
**Output Type(s):** [Shell commands, YAML configuration, Analysis] <br>
**Output Format:** [Markdown with inline bash and YAML code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Writes AISim output directories where the agent is told to; installing aisim writes the Python environment and package cache; the optional spot-check deploys to the user's host or cluster after approval] <br>

## Evaluation Metrics Used: <br>
[Filled by NVSkills CI on signing.] <br>

## Skill Version(s): <br>
0.1.0 (draft; targets aisim 0.14.0; full evals pending) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
