Graduate Project
Introduction
The graduate project is offered only to students enrolled in CS289A. Other students are welcome to work on a final project, but their work will not be graded or counted towards their final grades.

Teammates
You must work in groups of two or three students. To give everyone experience collaborating on a machine learning project, individual projects are not allowed.

Project Topic
The purpose of the project is to give you hands-on experience with the full ML pipeline from start to finish, including problem definition, dataset curation, methodology design, model training, evaluation, iteration, and presentation. The project should build upon course concepts. You’re encouraged to propose ideas that align with your own interests, but do not propose a project you have already completed as part of your research or another course. Otherwise, we are eager to help you identify a reasonable project that is engaging and useful for your learning.

Every project must meet the following criteria:

Task definition: Come up with a well-defined task you want to solve.
Motivate why the task is important and interesting. Ideally, the task would align with your own interests.
Data curation: Obtain appropriate training and testing datasets for your task.
For example, you can find existing datasets, scrape the internet, or synthesize data with generative models.
Methodology: Try different approaches. You must include approaches that perform gradient descent via back propagation (i.e., neural networks). You should rigorously evaluate across the design space for your task.
For example, you can try different: types of models, features and engineered features to include, training data curation methodologies, hyperparameters, etc.
Evaluation: Justify the design decisions that led to your final model and contextualize it relative to existing methods in the field.
Discuss tradeoffs and findings across the different approaches you tried.
Justify your final design decisions through ablation studies.
Compare your final performance against established methods.
There are two paths available that differ only in expectations around methodology.

Path 1 is for those who want to post-train a foundation model. Here, we do not expect you to try different types of models, but instead expect deeper study of what makes post-training work best on your chosen model, including features, data, hyperparameters, etc. Some starter ideas for inspiration:

Post-train a language model to mimic your writing.
Post-train a vision model to caption images.
Use foundation models to predict market changes based on news
Replicate existing post-training work on a new task, e.g. STaR, SDFT, Advisor Models, etc.
Path 2 is more open-ended and allows you to select a task and methodology of your choice. Here, we expect to see different types of models used, and for you to compare performance and compute requirements across them. Some starter ideas for inspiration:

Classification, generation, or segmentation of medical data
Image segmentation on a driving dataset
Anomaly detection in financial data
Milestones and Grading Breakdown
Deliverable	Due Date (11:59 PM PT)	Grading Weight
Project Proposal	Fri, Oct 9	5%
Progress Report	Fri, Nov 6	10%
First Draft of Paper	Tue, Nov 24	20%
Peer Review	Fri, Dec 4	10%
Final Report	Fri, Dec 11	40%
Final Video	Fri, Dec 11	15%
Project Proposal
Each group will submit a two-page project proposal, detailing the scope of their project and a rough timeline of what they plan to accomplish. You must get your project proposal approved by a professor or GSI before submitting it on Pensive. To seek approval, you may present a printed copy of your proposal during instructor/GSI office hours or email cs189-instructors@berkeley.edu. Office hours are also a great place to brainstorm and get feedback on your project ideas. To receive full credit, your proposal should contain the following:

Background:
What is the application domain or field of research?
What is the overarching goal of your project?
Data/Resources:
What data do you intend to work with?
Describe the features, labels, number of samples, etc.
Is this data publicly available? How will you access it?
Are there any other resources needed for your project (e.g., compute)? Do you have access to these?
Timeline:
How will each team member contribute to the project?
What do you plan to accomplish each week? Include internal milestones in addition to our required ones.
Progress Report
Each group will submit a 1-2 page report of what they have accomplished and their plans for the remainder of the project. To receive full credit, your progress report should include the following:

Outline:
Before submitting the progress report, each group should create a shared document to be used for their final report. You are encouraged to use Overleaf Professional, but Google Docs or any other method for collaborative writing is fine.
Within this document, create an outline for your report. Please see the section below on the “First Draft of Paper” for expectations regarding what is to be included in your report.
Provide a shareable link to this document in your progress report.
Progress:
Students should demonstrate substantial progress towards the goals outlined in their original project proposal.
Significant deviations from the original proposal should be discussed with course staff before submitting the progress report.
Describe where you are in your proposed timeline. Include a summary of what each team member has contributed and any preliminary results.
You are encouraged to share your code (preferably with a link to a GitHub repo) to demonstrate your progress.
Goals:
Include an updated timeline for the remainder of your project, following the guidelines for the project proposal.
First Draft of Paper
Each group will submit the first draft of their paper to be reviewed by other teams. This should be a refined paper that has already gone through internal review within the group. You must follow the ICML paper guidelines for your submission (except we don’t require double-blind reviewing). You are expected to adhere to the style guidelines and page requirements. Please note that we will not grade submissions that are longer than the maximum 8 pages. To receive full credit, your paper should contain the following:

Abstract – A complete summary of your paper.
Related Work – Discussion of other papers related to your own.
[Optional] Background – Additional information needed to understand your work.
Methods/Approach – A thorough description of what you did.
Evaluation/Analysis – Discussion of your experiments and visualizations of your results.
Conclusions – Summary of the key takeaways, as well as a discussion of limitations and areas for future work.
You do not need to follow this structure exactly and can use section headings/subheadings that make the most sense for your work, but all of this information should be captured in your paper to receive full credit.

Peer Review
Each student will review the paper of one other group and submit a one-page review, following the ICML 2025 reviewer guidelines, specifically the 2025 Main Track Reviewer Form Instructions. (Note that we will be following the ICML 2025 guidelines, not the ICML 2026 guidelines.) To receive full credit, you should answer the questions under:

Summary
Claims and Evidence
Relation to Prior Works
Other Aspects
Questions for Authors
You can ignore the sections on Ethical Issues, Code of Conduct Acknowledgement, and Overall Recommendation.

Final Report
Each group will revise their paper based on all the reviews they received and submit a polished version of their final report. This report should follow the same guidelines discussed for the First Draft of Paper.

In addition to your final report, you must submit a brief response to your reviewers, addressing reviewers’ comments and questions. To receive full credit on your final report, you should clearly indicate how you responded to reviewer feedback, including changes you made and those you chose not to make.

Final Video
Along with the final report, each group will submit a 2-3 minute video summarizing their project. To receive full credit:

The video should be clear and understandable, describing everything you think is important about your project (motivation, background, techniques, results, etc.).
The video needs to be self-contained: any CS 289A student should be able to understand what you did (at least at a high level) without consulting any other materials.
The video can be at most 3 minutes long. Note that this is a strict requirement; we will not grade a video that is more than 3 minutes long.
There is no requirement on the format of your video. You can make the video as simple as slides with a voice overlay or as fancy as you want.
Project Policies
AI Usage Policy
You may use AI tools to brainstorm ideas, write code, debug, find related work, summarize existing work, etc. However, you are fully responsible for your own work, and all of the written work must be entirely written by you, including the peer review. Please note that GenAI tools, like ChatGPT, tend to hallucinate when finding and citing related work. You must ensure that your paper is not in violation of UC Berkeley’s Academic Integrity policy.

Late Policy
The project proposal must be submitted on time to ensure that you receive timely feedback and have an appropriate project for this course. To enable a smooth peer review process, the first draft and peer review feedback must also be submitted on time. We are unlikely to grant extensions on any of these milestones, so please contact us only for exceptional circumstances.

All other milestones (the progress report, final report, and final video) will incur a 10% daily penalty after the due date, up to a maximum of two days. Submissions are rounded up to the next day (e.g., 2 minutes late counts as 1 day late).