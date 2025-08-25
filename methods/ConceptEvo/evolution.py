from google import genai
import random
import time

def concepts_init(client):
    """
    Function to initialize key concepts for concept evolution.
    
    Returns:
        List of key concepts.
    """

    video_file_name = "/home/sharing/disk1/Datasets/MIntRec/raw_data/S05/E13/529.mp4"
    video_bytes = open(video_file_name, 'rb').read()

    history = []
    history.append({
        "role": "user",
        "parts": [
                    {
                        "inline_data": {
                            "mime_type": "video/mp4",
                            "data": video_bytes
                        }
                    },
                    {
                    "text": "You are an expert in multimodal interaction analysis. Multimodal intent understanding aims to capture \
                    the speaker's intent based on the video and the spoken text. The possible intent list is ['Complain', 'Praise', \
                    'Apologise', 'Thank', 'Criticize', 'Agree', 'Taunt', 'Flaunt', 'Joke', 'Oppose', 'Comfort', 'Care', 'Inform', \
                    'Advise', 'Arrange', 'Introduce', 'Leave', 'Prevent', 'Greet', 'Ask for help']. \
                    Take the video with text (so your new desk is gonna be here in three days.) as an example, the intent can be \
                    inferred from various key concepts such as facial expressions, body movements, tone of voice, volume, specific \
                    intent words, and so on. Please analyze what key concepts are essential to the multimodal intent recognition task, \
                    and list the top 30 most important concepts that would help in understanding speaker intent across different scenarios.\
                    The concepts should be semantically meaningful, discriminative, and broadly applicable across multiple intent categories.\
                    Avoid concepts that simply duplicate intent labels. Respond only with the list of key concepts, separated by commas."
                    }
                ]
    })

    response = "Facial Expression, Tone of Voice, Body Language, Gaze Direction, Word Choice, Sentence Structure, Contextual Clues, Speaker's Role, \
                Addressee's Role, Relationship Dynamics, Politeness Markers, Intensity, Formality, Directness, Emotional State, Certainty, Hesitation, \
                Humor, Irony, Sarcasm, Emphasis, Topic of Discussion, Prior Interactions, Cultural Norms, Background Noise, Visual Cues, Pauses, Interruptions, \
                Speech Rate, Volume."
    history.append({
        "role": "model",
        "parts": [{"text": response}]
    })
    key_concepts = [c.strip().lower() for c in response.rstrip('.').split(',') if c.strip()]
    return key_concepts, history

    # response = client.models.generate_content(
    #     model='models/gemini-2.0-flash',
    #     contents=history
    # )
    # history.append({
    #     "role": "model",
    #     "parts": [{"text": response.text}]
    # })
    # key_concepts = [c.strip().lower() for c in response。text.rstrip('.').split(',') if c.strip()]
    # return key_concepts, history


def concepts_evolution(client, history, concept, D_score, S_score, epoch_num, loss, eval_score):
    """
    Function to perform concept evolution in a multimodal intent model.
    
    Args:
        model: The multimodal intent model.
        data_loader: DataLoader for the training data.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device: Device to run the model on (CPU or GPU).
    
    Returns:
        None
    """
    concept_lines = [f"{name}: D_score = {d_score:.4f}, S_score = {s_score:.4f}" for name, d_score, s_score in zip(concept, D_score, S_score)]
    # concept_lines = [f"{name}: S_score = {s_score:.4f}" for name, s_score in zip(concept, S_score)]
    concept_info = "\n".join(concept_lines)
    print(concept_info)

    history.append({
        "role": "user",
        "parts": [
                {
                "text": f"""
                    You are an assistant for training a multimodal intent recognition model that relies on a set of semantic concepts to enhance interpretability and performance.

                    At this stage, you are provided with:
                    - The current list of concepts, their associated discriminability scores (D_score), where **higher D_score indicates stronger discriminative ability** across intent categories, and their similarity scores (S_score), where higher values indicate stronger semantic overlap with other concepts.
                    - The current epoch number, training loss, and validation performance score.
                    - The goal is to iteratively refine the concept list by removing weak or redundant concepts and introducing more effective ones.

                    **Instructions:**
                    1. Review the concepts and their D_scores.
                    2. Identify the **top 5 most discriminative concepts** (highest D_scores) and analyze their common semantic or structural characteristics.
                    3. Use these shared characteristics to **propose new concepts** that aligns well with them.
                    4. Replace the **least discriminative concept** (least D_score) and  with new concept.
                    5. Replace the **highest similar concept** (highest S_score) and  with new concept.
                    6. Ensure the total number of concepts remains **unchanged**.

                    **Training Status:**
                    - Epoch: {epoch_num:.4f}
                    - Training Loss: {loss:.4f}
                    - Validation Score: {eval_score:.4f}

                    **Concepts and Scores:**
                    {concept_info}

                    **Output Format:**
                    Return **only** the updated concept list as a comma-separated string. **Do not include explanations or rankings.**
                    """
                # "text": f"""Now you are assisting in the training of a multimodal intent recognition model. At this stage, you are provided with the discriminability scores \
                # for each concpte, the current epoch's training loss, validation performance, and the historical context of past concept selections and their impacts. \
                # Your task is to refine the current concept list based on the following information: \
                # - Concepts with **lower D_score values** are more discriminative and should be prioritized. \
                # - The goal is to optimize for a set of analytical concepts that are both **effective and generalizable** across multiple intent categories. \
                # - Please avoid concepts that are redundant or simply duplicate intent label names. \
                # Training status: \
                # - Epoch: {epoch_num:.4f}  \
                # - Training Loss: {loss:.4f} \ 
                # - Validation Score: {eval_score:.4f} \
                # Here are the current concepts and their corresponding discriminability scores (D_score): {concept_info} \
                # Please analyze the top 5 concepts with the highest learning scores to identify their common characteristics. Based on this, infer an improved concept that \
                # could replace the bottom concept with the lowest scores. Keep the total number of concepts unchanged. Return only the updated concept list, formatted as a comma-separated string. Do not include any explanation."""
                }
            ]
        # "parts": [
        #             {
        #             "text": f"""Now you are assisting in the training of a multimodal intent recognition model. At this stage, you are provided with the current epoch's \
        #             training loss, validation performance, and historical context of past concept selections and their impacts. Your task is to analyze these scores and \
        #             signals to refine the current concept list. The goal is to optimize for a set of effective and generalizable analytical dimensions that support model \
        #             learning and contribute to improved performance. The epoch number is {epoch_num:.4f}, the loss is {loss:.4f}, and the eval score is {eval_score:.4f}. Please return the updated list of concepts , ranked \
        #             by importance. Avoid concepts that simply duplicate intent labels. Respond only with the list of key concepts, separated by commas."""
        #             }
        #         ]
    })
    response = safe_generate_content(
        client=client,
        model='models/gemini-2.0-flash',
        history=history
    )
    # response = client.models.generate_content(
    #     model='models/gemini-2.0-flash',
    #     contents=history
    # )
    if response is None:
        print("[INFO] Concept list unchanged due to API failure.")
        return concept, history
    
    history.append({
        "role": "model",
        "parts": [{"text": response.text}]
    })
    key_concepts = [c.strip().lower() for c in response.text.rstrip('.').split(',') if c.strip()]
    return key_concepts, history



def safe_generate_content(client, model, history, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=history
            )
            return response
        except Exception as e:
            # 如果是临时性错误，尝试重试
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"[Retrying in {sleep_time:.2f}s] Model busy or unavailable, attempt {attempt + 1}/{max_retries}")
                time.sleep(sleep_time)
            else:
                print(f"[ERROR] Non-retryable error encountered: {e}")
                return None
    print(f"[WARNING] Failed after {max_retries} retries. Keeping previous concept list.")
    return None

# "parts": [
#             {
#             "text": f"""Now you are assisting in the training of a multimodal intent recognition model. At this stage, you are provided with the discriminability scores \
#             for each concpte, the current epoch's training loss, validation performance, and the historical context of past concept selections and their impacts. \
#             Your task is to refine the current concept list based on the following information: \
#             - Concepts with **lower D_score values** are more discriminative and should be prioritized. \
#             - The goal is to optimize for a set of analytical concepts that are both **effective and generalizable** across multiple intent categories. \
#             - Please avoid concepts that are redundant or simply duplicate intent label names. \
#             Training status: \
#             - Epoch: {epoch_num:.4f}  \
#             - Training Loss: {loss:.4f} \ 
#             - Validation Score: {eval_score:.4f} \
#             Here are the current concepts and their corresponding discriminability scores (D_score): {concept_info} \
#             Please return the updated concept list, and formatted as a comma-separated string. Do not include any explanation."""
#             }
#         ]