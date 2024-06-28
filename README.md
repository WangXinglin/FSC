# Integrate the Essence and Eliminate the Dross: Fine-Grained Self-Consistency for Free-Form Language Generation [ACL2024 Main]

<div align="center">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/fdba812b-4722-44fd-b195-e6d181a8465f" width="700">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/17289605-b543-4b2f-9872-ca976c864bc9" width="700">
</div>


**Self-consistency (SC), leveraging multiple samples from LLMs, shows significant gains on various reasoning tasks but struggles with freeform generation due to the difficulty of aggregating answers. Its variants, UCS and USC, rely on sample selection or voting mechanisms to improve output quality. These methods, however, face limitations due to their inability to fully utilize the nuanced consensus knowledge present within multiple candidate samples, often resulting in suboptimal outputs. We propose Fine-Grained Self-Consistency (FSC) to addresses these limitations by extracting and integrating segment-level commonalities from candidate samples, enhancing the performance of LLMs both in open-ended and reasoning tasks. Based on this, we present two additional strategies: candidate filtering, which enhances overall quality by identifying highly similar candidate sets, and merging, which reduces input token requirements by combining similar samples. The effectiveness of FSC is demonstrated through extensive experiments on various tasks, including summarization, code generation, and mathematical reasoning, using GPT3.5-turbo and GPT-4. The results indicate significant improvements over baseline methods, showcasing the potential of FSC to optimize output quality by effectively synthesizing finegrained consensus knowledge from multiple samples.**


More details for the use of code will be coming up soon.

## Results

<div align="center">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/49a6e53f-6293-4ef5-a022-06a414b94465" width="700">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/20227bdb-175a-4375-9532-2296883e200f" width="700">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/585abc16-c45e-493a-a3a4-24f3717ef37e" width="350">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/d60951f9-59f3-4934-9946-3abf13af5eac" width="350">
    <img src="https://github.com/WangXinglin/FSC/assets/54845942/c2a787cc-4dfe-4ea9-a9e9-06951beca55e" width="700">
</div>

