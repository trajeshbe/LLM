

# Using LangGraph and CogVideoX for Image-to-Video Generation in Google Colab



This guide demonstrates how to leverage **LLM** with  **LangGraph** Agents and the **CogVideoX** model for transforming static images into animated videos using natural language descriptions. The approach combines image-to-video generation with LangGraph for workflow automation and a dynamic AI agent-powered pipeline.



### Overview



In this article, we'll explore the use of **CogVideoX-5b-I2V** for generating animated videos from images, and **LangGraph** for automating the process by building a state-driven agent workflow. We'll go through setting up the necessary models, installing dependencies, and implementing an automated pipeline for image-to-video conversion, including dynamic animation descriptions and titles.



### Prerequisites



1. **Google Colab** with GPU support (used L4 GPU, free-tier GPUs such as T4 may experience memory issues).
2. Hugging Face account and API token to access pretrained models.
3. OPEN AI API Token.
4. The following Python libraries:

- `diffusers`
- `transformers`
- `langgraph`
- `langchain_openai`
- `python-dotenv`

### Step 1: Installing Necessary Requirements



Begin by installing the essential libraries for the notebook:



```bash
!pip install diffusers transformers hf_transfer

!pip install accelerate==0.33.0

!pip install langgraph

!pip install langchain langchain_openai

!pip install python-dotenv

```



These libraries provide the required tools to run the CogVideoX pipeline, load pretrained models, and handle language-based agent interactions.



### Step 2: Model Loading and Setup



Once the libraries are installed, we load the necessary pretrained models to set up the **CogVideoX** pipeline for image-to-video generation.



```python
import torch

from diffusers import AutoencoderKLCogVideoX, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel

from transformers import T5EncoderModel

from dotenv import load_dotenv



# Enable Hugging Face transfer acceleration

import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"



# Load models for the CogVideoX pipeline

model_id = "THUDM/CogVideoX-5b-I2V"



# Hugging Face login (provide your token)

!huggingface-cli login --token "put_your_huggingface_token"



transformer = CogVideoXTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.float16)

text_encoder = T5EncoderModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16)

vae = AutoencoderKLCogVideoX.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float16)



# Create the pipeline for image-to-video generation

pipe = CogVideoXImageToVideoPipeline.from_pretrained(

    model_id,

    text_encoder=text_encoder,

    transformer=transformer,

    vae=vae,

    torch_dtype=torch.float16,

)



# Enable sequential CPU offloading to save GPU memory

pipe.enable_sequential_cpu_offload()

```



**Note**: The `float16` data type is recommended over `bfloat16` to avoid out-of-memory errors on Turing GPUs.



**Step 3: Image-to-Video Generation**



****how to generate captions for images using the **BLIP (Bootstrapped Language-Image Pretraining)** model from Salesforce and process a directory of images. Here's a detailed breakdown of each component:



------



### **1. Import Libraries**



```

import os

from PIL import Image

from transformers import BlipProcessor, BlipForConditionalGeneration



```



- **os**: Handles file and directory operations (e.g., iterating through files in a directory).
- **PIL.Image**: Used to open and manipulate images in Python.
- **transformers**: Provides pre-trained models and processors. Here, it loads the BLIP model and processor for image captioning.

------



### **2. The describe_image Function**



#### Purpose:



Generates a caption/description for a given image using the BLIP model.



#### Steps:



```


# Load the model and processor from Hugging Face

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")



```



- **Processor**: Prepares the image for the model (e.g., resizing, normalization).
- **Model**: BLIP's caption generation model.

```

# Open the image and prepare it for processing

image = Image.open(image_path).convert('RGB')



```



- The image is loaded using `PIL.Image` and converted to RGB mode, ensuring compatibility with the model.

```

# Process the image and generate caption

inputs = processor(image, return_tensors="pt")

output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens=True)



```



- **Input Preparation**: Converts the image into tensor format suitable for the BLIP model.
- **Caption Generation**: The model processes the image and outputs a sequence of tokens (a caption).
- **Decoding**: Converts the tokenized output back into a human-readable string.





------



### **3. Directory Processing**



```

image_directory = '/content/sample_data/images'
for filename in os.listdir(image_directory):

    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):

        image_path = os.path.join(image_directory, filename)

        print(f"Processing image: {image_path}")



```



- **Directory Path**: The script processes all image files in the specified directory (`/content/sample_data/images`).
- **File Check**: Ensures only valid image files (with extensions like `.png`, `.jpg`) are processed.

------



### **4. Image Description Workflow**



```
python





Copy code

state = AgentState(image_path=image_path)



```



- A state object (`AgentState`) is initialized to hold the `image_path` and other metadata about the current image.

```
description = describe_image(image_path)

if description:

    print(f"Image description for {filename}: {description}")



```



- Calls `describe_image()` to generate a description.
- If a caption is successfully generated, it is printed.

------



### **5. Invoking the Workflow**



```
app.invoke({"image_path": image_path, "description": description})



```



- Passes the image path and description to the compiled workflow


- Save the generated caption to a database.
- Perform further actions like creating animations based on the description.

------

### **How It Works**



1. The code scans the directory (`image_directory`) for images.
2. For each valid image file:

- Opens the image.
- Processes it with the BLIP model to generate a caption.
- Passes the caption and image path to the LangGraph workflow (`app.invoke()`).

1. If errors occur during description generation or file processing, they are logged, and the script skips to the next file.

------



### **Use Cases**



- **Image Captioning**: Automatically generating captions for a set of images.
- **Input for Other Workflows**: Providing captions as input to other tools (e.g., creating animations or descriptions for a catalog).
- **Dataset Annotation**: Automatically labeling images with textual descriptions.

### Step 3: Image-to-Video Generation



Now that we have the pipeline ready, we can proceed to generating animated videos from static images. You can supply a prompt and an image, and the pipeline will generate a video based on the description provided.



```python
# Example: Set up an animation description and image

image = load_image("/content/sample_data/images/IMG-20241027-WA0057.jpg")

prompt = "An astronaut hatching from an egg on the moon, with space in the background."



# Generate the video

video = pipe(image=image, prompt=prompt, guidance_scale=6, use_dynamic_cfg=True, num_inference_steps=50).frames[0]



# Export the video

export_to_video(video, "astronaut_hatching.mp4", fps=8)

```



This code generates an animated video from a static image by interpreting the prompt and applying transformations to simulate motion.



### Step 4: LangGraph Agents for Automation



We now introduce **LangGraph**, a library that enables the automation of complex workflows via state-driven agents. This allows you to build a sequence of tasks such as generating animation descriptions, creating titles for videos, and finally generating and exporting the video. LangGraph's agents manage state and can interact with language models (like OpenAI's GPT) to automate these processes.



#### Step 4.1: Define LangGraph Agent State



We'll define an agent state class to store image paths, descriptions, animation instructions, and video titles.



```python
from typing import TypedDict



class AgentState(TypedDict):

    image_path: str

    description: str

    desc_for_animation: str

    tilte_for_video: str

```



#### Step 4.2: Implement Workflow Functions



We will create functions that:

1. Generate an animation description based on the image's description.
2. Create a title for the video.
3. Generate the animation and export it.

```python
from langchain_openai import ChatOpenAI



# Initialize the OpenAI model

model = ChatOpenAI(temperature=0)



def how_to_animate_desc(state: AgentState):

    description = state.get("description")

    complete_query = f"Provide instructions for an animation in 15 words: {description}"

    response = model.invoke(complete_query)

    state["desc_for_animation"] = response.content

    return state



def title_for_video(state: AgentState):

    description = state.get("description")

    complete_query = f"Generate a title for the video: {description}"

    response = model.invoke(complete_query)

    state["tilte_for_video"] = response.content

    return state



def create_animation(state: AgentState):

    desc_for_animation = state.get("desc_for_animation")

    image_path = state.get("image_path")

    image = load_image(image_path)

    

    video = pipe(image=image, prompt=desc_for_animation, guidance_scale=6, num_inference_steps=50).frames[0]

    

    output_path = f"/content/sample_data/output_videos/{state['tilte_for_video']}.mp4"

    export_to_video(video, output_path, fps=8)

    return state

```



#### Step 4.3: Define the Workflow



Next, we'll define a LangGraph `StateGraph` that connects the above steps into a single workflow.



```python
from langgraph.graph import StateGraph



# Initialize the workflow

workflow = StateGraph(AgentState)



# Add nodes for each step in the process

workflow.add_node("Describe_Animation", how_to_animate_desc)

workflow.add_node("Create_Title", title_for_video)

workflow.add_node("Create_Animation", create_animation)



# Define the order of execution

workflow.add_edge("Describe_Animation", "Create_Title")

workflow.add_edge("Create_Title", "Create_Animation")



# Set the entry and finish points

workflow.set_entry_point("Describe_Animation")

workflow.set_finish_point("Create_Animation")



# Compile the workflow

app = workflow.compile()

```



#### Step 4.4: Run the Workflow



To invoke the workflow, we pass an image and its description to the LangGraph app, and the app will automatically generate the video.



```python
image_path = '/content/sample_data/images/IMG-20241027-WA0057.jpg'  # Replace with your image path

description = describe_image(image_path)  # Generate description using BLIP model



# Invoke the workflow

app.invoke({"image_path": image_path, "description": description})

```



### Step 5: Visualizing the Workflow



To visualize the workflow, you can generate a Mermaid diagram of the LangGraph state transitions.



```python
from IPython.display import Image, display



# Display the workflow graph

display(Image(app.get_graph().draw_mermaid_png()))





```



### Example: Image-to-Video Workflow for **IMG20241020200449.jpg**



#### 1. **Image Description Generation**



- **Input Image**: `/content/sample_data/images/IMG20241020200449.jpg`
- **Generated Image Description**:

  *"A painting of a rainbow and butterflies"*



#### 2. **How to Animate the Image**:



- In `how_to_animate_desc`

  :



- The agent receives the image description:

    *"A painting of a rainbow and butterflies"*

- The agent generates the animation description:

    ***Animate butterflies fluttering around a rainbow, with joyful expressions on their faces.\***

- **Output**: The animation description is stored in the state for later use.

#### 3. **Video Title Creation**:



- In `title_for_video`

  :



- The agent generates a title based on the image description:

    ***Rainbow_and_Butterflies_Painting\***

- **Output**: The title is stored in the state for use when exporting the video.

#### 4. **Creating the Animation**:



- In `create_animation`

  :



- The agent proceeds to create the animation based on:


- **Description**: *"Animate butterflies fluttering around a rainbow, with joyful expressions on their faces."*
- **Title**: *"Rainbow_and_Butterflies_Painting"*
- **Image Path**: `/content/sample_data/images/IMG20241020200449.jpg`


- The agent processes the image and animation description with **CogVideoX**, and begins generating the animation.


- Animation Process

    :



- The agent starts the animation with the description: *"Animate butterflies fluttering around a rainbow, with joyful expressions on their faces."*
- The **video** is generated and exported.


- **Export Path**: `/content/sample_data/output_videos/Rainbow_and_Butterflies_Painting.mp4`

#### 5. **Final Output**:



- **Exported Video**:

  The video is created and saved as:

  `/content/sample_data/output_videos/Rainbow_and_Butterflies_Painting.mp4`



------



### Visual Breakdown of the Process



```
plaintextCopy codeImage Description for IMG20241020200449.jpg: a painting of a rainbow and butterflies



1. In how_to_animate_desc:

   Agent Says how_to_animate: Animate butterflies fluttering around a rainbow, with joyful expressions on their faces.



1. Creating Title for Video:

   Agent Says the title is: Rainbow_and_Butterflies_Painting



1. Creating Animation:

   - Creating Animation - desc_for_animation: Animate butterflies fluttering around a rainbow, with joyful expressions on their faces.

   - Creating Animation - title_for_video: Rainbow_and_Butterflies_Painting

   - Creating Animation - image path: /content/sample_data/images/IMG20241020200449.jpg

   - Animation started for: Animate butterflies fluttering around a rainbow, with joyful expressions on their faces.

   - Exporting video to: /content/sample_data/output_videos/Rainbow_and_Butterflies_Painting.mp4

```



------



### Key Points:



1. **Image Description**: The image description helps the model understand the content and context of the image (in this case, a painting with butterflies and a rainbow).
2. **How to Animate**: The agent generates specific instructions for animating the scene, focusing on the action and emotion (e.g., "fluttering butterflies" and "joyful expressions").
3. **Video Title**: A descriptive title is created to label the generated video, which is based on the content of the image.
4. **Animation Generation**: The model takes the image and animation description to generate a video, which is exported with the generated title.

------



This example shows the full pipeline in action, from describing the image and generating an animation description to creating the video and saving it as an output file. The entire process is automated using **LangGraph agents**, which helps in managing the flow of data and automating complex tasks like video title generation and animation creation.



### Conclusion



This tutorial demonstrates how to combine the power of **CogVideoX** for image-to-video generation with **LangGraph** for workflow automation. By using LangGraph agents, you can streamline the process of generating animations from images, automating tasks like animation description creation, title generation, and video rendering. The result is an efficient and flexible pipeline for creative video production.

