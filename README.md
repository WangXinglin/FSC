# Integrate the Essence and Eliminate the Dross: Fine-Grained Self-Consistency for Free-Form Language Generation [ACL2024 Main]

<div align="center">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/32fe2dd3-bf77-422e-841a-11b40ae0f5ff" width="700">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/0f59b321-9317-407a-895a-48b736b3ee92" width="700">
</div>


**Self-consistency (SC), leveraging multiple samples from LLMs, shows significant gains on various reasoning tasks but struggles with freeform generation due to the difficulty of aggregating answers. Its variants, UCS and USC, rely on sample selection or voting mechanisms to improve output quality. These methods, however, face limitations due to their inability to fully utilize the nuanced consensus knowledge present within multiple candidate samples, often resulting in suboptimal outputs. We propose Fine-Grained Self-Consistency (FSC) to addresses these limitations by extracting and integrating segment-level commonalities from candidate samples, enhancing the performance of LLMs both in open-ended and reasoning tasks. Based on this, we present two additional strategies: candidate filtering, which enhances overall quality by identifying highly similar candidate sets, and merging, which reduces input token requirements by combining similar samples. The effectiveness of FSC is demonstrated through extensive experiments on various tasks, including summarization, code generation, and mathematical reasoning, using GPT3.5-turbo and GPT-4. The results indicate significant improvements over baseline methods, showcasing the potential of FSC to optimize output quality by effectively synthesizing finegrained consensus knowledge from multiple samples.**


More details for the use of code will be coming up soon.

## Results

<div align="center">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/6eb908dc-8d07-4371-842f-76e31893e32b" width="700">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/aaaab8ad-91dd-486f-814d-81c4d6d0b6a7" width="700">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/2fe66c5c-757a-4a4c-94d4-1b7a6c99e00c" width="350">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/5f688c10-97f6-42c1-8cd1-75cd1f713030" width="700">
</div>

