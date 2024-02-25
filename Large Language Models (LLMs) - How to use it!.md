### Large Language Models (LLMs) in Business — How to use it?

A Language model is anything that can predict the next word given a word.

Example:

**Jessie loves ice ___ ?**

Most likely we would predict the next word as “cream”. Despite we know few words like ice berg, ice land, ice cube, we choose “cream” because our experience, learning and knowledge about the world pushes us to pick cream as the most probable word. btw , If you have guessed something else, it’s not wrong either.

**Martin loves hockey. His favorite sport is ice _____?** .

Now given the context of words hockey, sports in the above sentence , we would most likely predict the next word as “hockey” rather than “cream” . This is all a language model does and if a large machine is able to do it at scale word after word, learn patterns, structures, and relationships within language, we call it Large Language Models (LLMs). Near human like results due to availability of data to train and computing power is what makes LLMs popular now.

The idea of generating the next word given a previous word based on probabilities exists for decades. What makes this good now is the quality of the output and performance. Lets now say Thanks to Transformers, an architecture which is the backbone that transformed this space .Things get outdated, like you blink and miss in Natural Language Processing space. Transformers lasting 8 odd years is longevity :). Mamba is the next promising one in line.

Lets come back to LLMs and how it can fit into your business or use cases of your organization, this will help to understand what it is and how it can be adopted. This is light on technical and should be easy to follow.

Let’s use an analogy of Hiring Resources in an organization with that of LLMs. Let’s say, we have three open positions in the organization. All we do is lookout for candidates with suitable skill sets & qualifications for different positions, conduct interviews, and then finally hire the right candidates. Certain positions are filled with college grads and certain ones with prior experiences based on the complexity and demands of the job profile.

#### Job Position 1: Fresh College Graduate (UG)

We have interviewed and hired fresh college graduate **Ram** with a Degree in Computer Science. The expectation is that he possess some knowledge which have been acquired through continuous learning all through his life, schooling, and graduation in the subjects of his major. Once hired, he is given orientation and training to obtain additional skills and knowledge which are project-specific or domain-specific. What we are necessarily doing is fine-tuning or can be said as adding a top-up to his existing knowledge to enable him for work.

#### Job Position 2: Post Graduate + 5 years work Experience

We have interviewed and hired **Rahim** with a Post Graduate in Computer Science + 5 years of work experience. The expectation is that he possess not just his postgraduate subjective knowledge, but also the additional knowledge they have gained over the years of experience working on various roles and projects. Most often, we don’t emphasize on training experienced hires with the same rigor that is shown for new college hires for obvious reasons.

#### Job Position 3: PhD + 10 years work experience

We have hired **Robert** with a Ph.D. Degree in Computer Science + 10 years of work experience. This is a highly specialized position and requires both vast and in-depth domain knowledge to carry out work.

Ram (UG), Rahim (PG), and Robert (PhD) have joined the team with the skills and knowledge they possess. All we have done is hired them for their pre-trained skills and knowledge.

Now let’s switch to LLMs. Just like Ram, Rahim and Robert possess different knowledge levels based on their background education and experience, LLMs also possess varying levels of knowledge and understanding based on the amount of training i.e Size, Quality, Diversity and representatives of the data, quality of teaching , time taken to train etc. Let’s now pick three LLMs

#### Zephyr 7B Beta (Just like Ram with UG),

#### Llama 2 70B (Rahim with PG),

#### ChatGPT 3.5 170B (Robert with PhD)

Note: This analogy is just for Intuition :).. All human brain more or less have similar number of neurons while these artificial neural network models have varying trainable parameters , be it 7, 70 170 or whatever the number may be!

Just like candidates , there are 1000s of available LLMs and we have picked few of them i.e interviewed and selected the above three of them . We can refer the leaderboard to pick from the possible optimal options which fit both our bill, constraints, limitations, and also reasonably fulfill expectations — [Chatbot Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)

As said, 7B, 70B, 170B are the Billions of parameters of the model. Just like degrees , bigger the number, the more the storage space they have to store the finite and precise knowledge they have learned and acquired. A Caveat, at times smaller ones (say 7B) can surprise the bigger ones(say 70B) on few tasks, just like nothing limits some exceptional junior folks who can provide innovate solutions.

In addition, 7B models are just like Toyota Corolla, you can just take it and run on any normal roads with less maintenance i.e can run on commodity hardware like NVIDIA Tesla T4 GPU. Bigger ones are like F1 cars, not easy to run or maintain on ordinary roads, need expensive memory & GPUs. Either way, looking at the latest financial results (Q4 FY24) of NVIDIA is making some good money :)

We also remind ourselves that Ram, Rahim, and Robert were raised by their parents or guardians, sent to school, paid for their school & college fees ,had great teachers to acquire knowledge. They also have spent considerable years of time in each phase of life learning stuff. All this is to highlight the fact that acquiring good knowledge is not easy task even for a human and much harder for a machine.

Various large corporates like Google (BARD, Gemma — opensource), Meta (Lamma2 Series — Open source), OpenAI (ChatGPT series — Proprietary), smaller ones like Anthropic’s (Claude — Proprietary), Mistral, Hugging face (Zephyr Series based on Mistral) are investing huge resources to train and create new and better state-of-the-art LLMs trying to outdo each other. So, if you are an organization trying to adopt one, you don’t have to train something from scratch and try to evaluate the one that suites your needs.

Proprietary ones (e.g ChatGPT) are like Greatly skilled External consultants that you hire and pay for work. They do a great job but at the same time, your organization isn’t free enough to provide additional trainings that are very specific to your organization to them OR share very sensitive resources to refer or fine-tune their knowledge based on your organization/domain-specific needs. You have to be sensitive about sharing business critical information with them as well. They work over the internet and they are a bit expensive as well. We still have trust issues on them. Nevertheless, ChatGPT 4 is still the best one in the business as on date.

Open-source ones (like Llama 2, Mistral, Zephyr) — they are like candidates who are near equally skilled but can be hired and put to learn your organization specific needs, flexible enough to get the job done in an organization. You can have them on-premise or in your cloud and can share your company-specific data for reference learning. They are relatively cheaper as well.

Now let’s focus on the chosen LLM candidate — Zephyr 7B Beta (remember, it’s like Ram with UG Degree) that we have hired . [Zephyr 7B Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) , a even lighter version of the same is named as <https://huggingface.co/TheBloke/zephyr-7B-beta-GPTQ> . you may notice something like GPTQ, GGUF or AWQ at the end of the models . All these are methods of Quantization, just like various means to lose some additional weight so that models can run even faster. on machine. While getting lighter is a good , quantization also loses some precision but at acceptable levels. **The Bloke** who helps us get access to these light weight models is like Usain bolt , they keep quantizing newer popular models as soon as they arrive to market.

GPTQ is a post-training quantization (PTQ) method for 4-bit quantization that focuses primarily on **GPU** inference and performance.

GGUF, previously GGML, is a quantization method that allows users to use the **CPU**.

A new format on the block is AWQ ([Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)) which is a quantization method similar to GPTQ.

While transformers overcome the drawbacks of sequential training of Recurrent Neural Networks, during inference (i.e. real time predictions) its bit slow. 

Ok, lets come back to our new hire Mr Zephyr. We wanted to give him a simple task and see how he learns and responds. The task is to answer some general questions and then ask more specific questions which he may or may not know. If he doesn’t know the answer, give some training and then ask the questions.

This process of expecting LLMs to answer the question using the existing knowledge is called using a pre-trained or foundation model i.e Expecting to answer questions OOTB with his existing knowledge. Zephyr-7b-beta is one of the best open-source alternatives to ChatGPT in many tasks, particularly Text Summarization, Text Understanding. What you see below is a performance benchmark just like GPA Scores of college grads.

Let’s say the pretrained model has answered a question wrong, what we now want is to learn again and then attempt to give the right answer. The process of expecting LLMs to answer the question after giving some instructions is called fine-tuning/supervised fine-tuning/instruction fine-tuning. i.e top-up its knowledge and try if it can get the answer right. In our case, we take pretrained Zephyr-7b-beta, give some sample Questions and Answers aka Instruction dataset, make it learn and then ask the question to the updated model and hope it gives the right answers.

Practically, we can’t keep spending time on training resources all the time if he doesn’t know an answer. i.e on Job if a resource doesn’t know how to do certain stuff, we don’t immediately arrange a 2-day training to learn and come back. if so, he has to be on training most of his job time :)

In such cases, all we do is give some quick references to documents, knowledge bases, some websites, which he can quickly refer and then answer the questions on the fly with the reference context. The process of converting the questions, the reference documents into a machine-readable, understandable , context-aware representation is called word embeddings. It’s stored in an embedding database, also known as a vector database. The process of quickly looking up the question or query with the reference documents in the vector database is called AI Search/AI Retrieval. The process of having the LLMs answering the question with the retrieved additional Information is called Retrieval Augmented Generation (RAG).

Now let’s move on, let’s ask some questions to Zephyr.

### Question to Zephyr:

**What is Newton’s third law?**Answer: Newton’s third law states that for every action, there is an equal and opposite reaction. This law applies to all forces that occur in nature, whether they are between two objects or within a single object.

Good job,

Let’s move on. Let’s assume our company has a weird HR policy. It has a short storybook which contains Part 1 and Part 2. Every new hire asked questions from it. Well, now the chances of the new hire to know this company specific story is very slim. So mostly he will not answer it right the first time, i.e just by using the pre-trained knowledge.

We haven’t yet revealed the location of the storybook yet, but still went ahead and asked the following question.

It’s a fictional story. Set up a plot as a prompt and allowed ChatGPT to expand the story. The Characters’ names in the story will be in native Tamil language. This to ensure that our pre-trained LLM candidate Mr Zephyr will have less clue about what is the context and test him well :)

Here is the location of the Story for Our reference (Zephyr has not yet seen it)

Please read the story if you could before proceeding further, it would help to get the context : [Short Story](https://github.com/trajeshbe/LLM/blob/main/Short%20Story.markdown)

### Question to Zephyr:

**Who is Aadhan?**

Answer: Aadhan is a popular Indian playback singer and composer who has lent his voice to several hit songs in Tamil, Telugu,

Hmm.. what’s happening is that Zephyr, like a Human will attempt to tell whatever we know as an answer to a question. We may be close enough or incorrect. This process of giving a factually incorrect answer is called hallucination. In this case above, being an Indian who lives in the region, haven’t heard of this playback singer. He might probably exist or probably not, but the point is in the context of our story, the Answer is wrong.

Hmm. so the next step is to give him the Part 1 of the story as Question and Answer. Just like we give students some handbook of Q&A to prepare for the exam, we give the same to Mr Zephyr and he reads and learns it.

Here is the handbook — also known as instruction set. Please go through it just to get the context : [Simple Story Question Answers for Finetuning](https://github.com/trajeshbe/LLM/blob/main/simple_story_question_answers_for_finetuning.csv)

Here is the code which is used to Fine tune the model with the Instruction Set above: [FineTuning with Simple Story](https://github.com/trajeshbe/LLM/blob/main/Zephyr_7B_FineTuning_custom_data_using_a_simple_story.ipynb)

Now the fine-tuning is complete and let’s ask the same question to the updated model, i.e the model which has now updated its knowledge with the additional training.

Mr Zephyr, a question for you: “Who is Aadhan?”

Answer: King’s son and Thamarai’s lover

Phew, the answer still looks to be incorrect but better than the pre-trained one. One of the reasons is in the Fine-tuning dataset, there wasn’t a direct question Who is Aadhan and a direct answer for it. So, the model attempts to reason but got the relationship wrong. You can check the data once again here — [FineTuning Data](https://github.com/trajeshbe/LLM/blob/main/simple_story_question_answers_for_finetuning.csv)

Let’s continue to ask Zephyr a few more questions.

**Question:** Who was Aadhan’s wicked brother?

**Answer:** Amudhan

Good, it got it right.

Few more questions.

**Question:** What is the name of the Kingdom of Aadhan?

**Answer:** Kandigai

You can look into the code and the results as well [here](https://github.com/trajeshbe/LLM/blob/main/Zephyr_7B_FineTuning_custom_data_using_a_simple_story.ipynb)

Good stuff, is the updated model has learned or memorized the data we have given as reference.

Mr. Zephyr can now answer the questions even if woken up at midnight :)

But there is still some problem, the Answer to Who is Aadhan is still not quite right. In addition, we have not yet given the Part 2 of the story. It can’t answer most of the questions from data which it has not seen yet. We got to do something better , as we can’t keep training again and again. It’s expensive and time consuming as well.

So, let’s use the next method to Vectorize all the reference documents ie the Q&A Dataset of part 1 & 2 of the story , store it in a vector DB. Pull the reference info on demand as context and then ask the question to Mr. Zephyr. This process is known as Retrieval Augmented Generations (RAG).

Q&A for the extended story for Part 2:** [Extended Story Question Answers for RAG](https://github.com/trajeshbe/LLM/blob/main/simple_extended_story_question_answers_for_rag.csv)

**Code for RAG:** [RAG Code](https://github.com/trajeshbe/LLM/blob/main/Zephyr_7B_Retrieval_Augmented_Generation_using_a_simple_story.ipynb)

### Lets continue the Q & A after Implementing RAG

**Question:** Who is Aadhan?

**Context:** The context was Document(page_content=’who is Aadhan?,The Emperor of Kandigai\nWho is Aadhan?,Aadhan is a noble king.

**Answer:** Aadhan is a noble king, as mentioned in the given context.

Wow.. it got it Right this time because, it has used both Part 1 & Part 2 Q&A as reference and pulled the closest question and Answer as a context for the query (Who is Aadhan?) and then tried to answer.

Now, you must be wondering, all we have been doing so far is either memorizing OR retrieving information with references . We are yet to see the Generative power of LLMs and allow it to respond beyond context. Let’s continue to ask a few more questions.

**Question:** What would be your advice for Amudhan?

**Answer:** Based on the given context, it seems that Amudhan’s actions as a ruler have led to hardships for his citizens. My advice for him would be to reevaluate his leadership style and prioritize the needs and well-being of his people over his own desires. As a tyrant, he may need to learn to govern with fairness, justice, and compassion. It is also clear from the text that Aadhan faced challenges in confronting Amudhan. Therefore, I suggest that Amudhan should consider listening to Aadhan’s perspective and working together towards a more just and equitable society. Ultimately, Amudhan must recognize that true power lies not in oppression but in serving the interests of his people.

Good, the response above shows that Mr. Zephyr has understood the character of Amudhan the villain and gives a piece of advice as well. Note that there wasn’t any questionnaire in the additional training data about advice or reforming Amudhan. Mr. Zephyr, the LLM, has generated these responses with some understanding of the textual world.

**Question:** What was the situation when Vedhan and Maria met for the first time?

**Context:** Document(page_content=’What did Vedhan and Maria exchange throughout their travels?, Cultural traditions and experiences’, metadata={‘source’: ‘/content/sample_data/simple_extended_story_question_answers_for_rag.csv’

**Answer:** Vedhan and Maria did not meet for the first time in the given context. The text describes how Vedhan saved Maria from a bear while she was being chased by it during her trekking.

Here Mr. Zephyr struggles to answer the question as it’s unable to make sense from the closest context chosen. Because our questionnaire doesn’t have enough references on how they met.

To overcome this, we can provide additional extracts from the story as a reference and then allow it to answer.

We have used the text stored as embeddings in the vector DB object (db.as_retriever()) so far for pulling out the context. Let’s add new context text and store it in another vector DB (vectorstore)

All we do is give additional information from the story and allow Mr. Zephyr to refer to it. Below is the piece of code that does it.

Converts our story into embeddings and stores it in a vector DB vectorstore

```
vectorstore=FAISS.from_texts(
    ["As time passed by Aadhan and Thamarai's son Vendhan, a prince turned 20. He was wise and disciplined wanted to explore the world. In the meantime, a trader from Portugal Cristiano Ronaldo visited the hills of Kandigai in search of some spices like Cardamom and pepper. He was with his family i.e his wife Carolina and beautiful daughter Maria. They were trekking on the hills. The Prince Vendhan was on the hills too on his horse to visit his friend Cheran who lives on top of the hill. He heard a cry for help and rushed towards the voice. He noticed Maria running and being chased by a bear. The prince rushed on his horse and picked Maria and rescued her. Saving her life, there emerged a romance. They soon fell in love. Maria took Vendhan to Portugal for a tour and explored Portuguese culture."], embedding=embeddings)

```

retriever1 is set up to retrieve the relevant closest information from the vectorstore above

```
retriever1 = vectorstore.as_retriever()

```

llm_chain has the LLM Object of Mr. Zephyr, which takes the question as input and the context from the retriever above

```
rag_chain = (
 {"context": retriever1, "question": RunnablePassthrough()}
    | llm_chain
)

```

We now ask the actual question

```
pythonCopy code
result=rag_chain.invoke("Where did Vendhan and Maria meet?")

```

Pulls out and refers to the new context that we just provided

```
result['context']

[Document(page_content="As time passed by Aadhan and Thamarai's son Vendhan, a prince turned 20. He was wise and disciplined wanted to explore the world. In the mean time, a trader from Portugal Cristiano Ronaldo visited the hills of Kandigai in search of some spices like Cardamom and pepper. He was with his family i.e his wife Carolina and beautiful daughter Maria. They were trekking on the hills. The Prince Vendhan was on the hills too in his horse to visit his friend Cheran who lives on top of the hill. He heard a cry for help and rushed towards the voice. He noticed Maria running and being chased by a bear. The prince rushed in this horse and picked Maria and rescued her. Saving her life, there emerged a romance. They soon fell in love. Maria took Vendhan to Portugal for a tour and explored the Portuguese culture")]

```

Response to the question

```
print(result['text'])

Vendhan and Maria met while Vendhan was rescuing Maria from being chased by a bear in the hills of Kandigai.

```

We can now see that it has read the context and answers correctly on where exactly they met each other.

Let’s continue with a few more questions before we wrap up.

**Question:** What was the situation when Vedhan and Maria met for the first time?**Answer:** Vedhan and Maria did not meet for the first time in the given context. The text describes how Vedhan saved Maria from a bear while she was being chased by it during her trekking

**Question:** Describe some features of celebration during Panguni Uthiram?**Answer:** During the Panguni Uthiram festival, elaborate processions and traditional dances are some of its prominent features. These celebrations are a part of the Tamil Hindu calendar and typically take place in March or April each year. The processions involve decorated floats carrying deities through the streets, accompanied by devotees singing hymns and chanting prayers. Traditional dance forms such as Bharatanatyam, Kavadi Attam, and Thiruvathira Kali are also performed during this festival, adding to the cultural richness of the celebrations. Overall, the Panguni Uthiram festival is a vibrant and colorful affair that showcases the deep-rooted traditions and beliefs of the Tamil community.

Now, from the response, let’s ask tougher questions to Mr. Zephyr . A dance form “Kavadi Attam” was mentioned in the response. It wasn’t part of our story.. Let’s push Mr. Zephyr to think beyond and let us know what it is?

**Question:** What is Kavadi Attam? Please answer even if it’s not part of the context”

**Answer:** I do not have prior knowledge or context outside of what is provided in the given text. However, based on my research, “Kavadi Attam” refers to a traditional Tamil festival that involves carrying a heavy structure called a kavadi as an act of penance and devotion to God Murugan. It is typically performed during the Thaipusam festival, which falls during the Tamil month of Thai (January-February).

Finally, we are done..

This completes the three methods to use LLM as

1. Want to use it Out of the Box? Use Pre-Trained
2. Want to use it with additional training? Use it with Fine Tuning
3. Want to use it OOTB with/Without additional training, but also with near real-time reference documents? Use Retrieval Augmented Generation (RAG) + OOTB Pretrained OR Retrieval Augmented Generation (RAG) + OOTB Pretrained with an added Fine-Tuned Model

Ok, now that we know LLMs can be smarter, faster, intelligent to some extent and promising does is really think and answer like Human? Answer is No. We are just not word/token generators :) .. Having said that, it has tremendous potential. It will and already would he replaced few jobs and will continue to do so, however it’s not yet ready for critical prod applications. Its super useful tool for Augmented  Data Labelling, Text Summarization , Text classification in an Enterprise and it can only grow better from here on.

Links to Colab Notebook to try and do it yourself. All you need is your GMAIL account, copy the code to your drive, and try it in Free T4 Instance (NVIDIA Tesla T4)

Fine-tuning — [FineTuning Colab](https://colab.research.google.com/drive/1feSTLzyftSsEGXkHt1R3UV_og367i1Xq?usp=sharing)

RAG — [RAG Colab](https://colab.research.google.com/drive/1-XNgAfViyKSqi__vfxxRaRpZr1R9Aubr?usp=sharing)

Get the two-story questionnaire CSV files from [here](https://github.com/trajeshbe/LLM) upload into your Colab -

/content/sample_data directory

Just Execute.

Note: You can replace Zephyr with other open source models, preferably quantized of your choice and it should most often work

Happy Learning!!

Thank you!

**References**

<https://medium.aiplanet.com/finetuning-using-zephyr-7b-quantized-model-on-a-custom-task-of-customer-support-chatbot-7f4fff56059d>

<https://www.maartengrootendorst.com/blog/improving-llms/>

<https://towardsdatascience.com/retrieval-augmented-generation-rag-from-theory-to-langchain-implementation-4e9bd5f6a4f2>

<https://www.e2enetworks.com/blog/zephyr-7b-beta-an-alternative-to-chatgpt>