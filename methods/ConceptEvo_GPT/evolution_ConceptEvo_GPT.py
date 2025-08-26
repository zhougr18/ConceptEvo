from openai import OpenAI
import random
import time
from textwrap import dedent

# Initialize the OpenAI client
client = OpenAI(api_key="your_api_key_here")

def concepts_init():
    """
    Function to initialize key concepts for concept evolution. The initial concept set is pre-generated and we keep both
        the prompt and its corresponding model response as historical records.
    """
    prompt = "You are an expert in multimodal interaction analysis. Multimodal intent understanding aims to capture \
                    the speaker's intent based on the video and the spoken text. The possible intent list is ['Complain', 'Praise', \
                    'Apologise', 'Thank', 'Criticize', 'Agree', 'Taunt', 'Flaunt', 'Joke', 'Oppose', 'Comfort', 'Care', 'Inform', \
                    'Advise', 'Arrange', 'Introduce', 'Leave', 'Prevent', 'Greet', 'Ask for help']. \
                    Take the video with text (so your new desk is gonna be here in three days.) as an example, the intent can be \
                    inferred from various key concepts such as facial expressions, body movements, tone of voice, volume, specific \
                    intent words, and so on. Please analyze what key concepts are essential to the multimodal intent recognition task, \
                    and list the top 30 most important concepts that would help in understanding speaker intent across different scenarios.\
                    The concepts should be semantically meaningful, discriminative, and broadly applicable across multiple intent categories.\
                    Avoid concepts that simply duplicate intent labels. Respond only with the list of key concepts, separated by commas."

    history = [{"role": "user", "content": prompt}]

    response = "Facial expression, Eye gaze direction, Eyebrow movement, Lip movement, Head nodding, Head shaking, Hand gestures, Posture changes, Body orientation, Proximity to listener, Tone of voice, Pitch variation, Speech rate, Volume intensity, Pauses in speech, Emphasis on words, Word choice, Prosody pattern, Smile presence, Frown presence, Eye rolling, Arm crossing, Shoulder shrugging, Tempo of movement, Gesture synchronization with speech, Gaze shifts, Intonation contour, Microexpressions, Hesitation markers, Repetition of words"


    history.append({"role": "assistant", "content": response})
    key_concepts = [c.strip().lower() for c in response.rstrip('.').split(',') if c.strip()]

    return key_concepts, history

def concepts_evolution(history, concept, D_score, S_score, epoch_num, loss, eval_score):
    """
    Concept evolution (using GPT-4), only keeping the most recent round of dialogue each time
    """
    # Clean up history: only keep the most recent round of conversation
    if len(history) > 2:
        # Preserve the system's initial settings (if any) and the most recent round of conversations
        if history[0]["role"] == "system":
            history = [history[0]] + history[-2:]
        else:
            history = history[-2:]
    
    # Build current evolution hint
    concept_lines = [
        f"{name}: D_score = {d_score:.4f}, S_score = {s_score:.4f}"
        for name, d_score, s_score in zip(concept, D_score, S_score)
    ]
    concept_info = "\n".join(concept_lines)

    prompt = f"""
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

    # Add current prompt to history
    history.append({"role": "user", "content": prompt})
    
    # Call the API to get a response
    response = safe_generate_content_gpt(history=history, model="gpt-4")
    
    if not response:
        print("[INFO] Concept list unchanged due to API failure.")
        return concept, history, concept_info
    
    # Add response to history
    history.append({"role": "assistant", "content": response})
    
    # Analyzing new concepts
    new_concepts = [c.strip().lower() for c in response.rstrip('.').split(',') if c.strip()]
    
    if len(new_concepts) != len(concept):
        print(f"[WARNING] Concept count mismatch. Expected {len(concept)}, got {len(new_concepts)}")
        return concept, history, concept_info
    
    # Clear the history again: only keep the most recent round of conversation
    if len(history) > 2:
        if history[0]["role"] == "system":
            history = [history[0]] + history[-2:]
        else:
            history = history[-2:]
    
    return new_concepts, history, concept_info

def safe_generate_content_gpt(history, model="gpt-4", max_retries=6):
    """
    Safely calling the GPT-4 API with retry mechanism
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=history,
                temperature=0.3,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            if "rate limit" in str(e).lower():
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"[Retry {attempt+1}/{max_retries}] Rate limit hit, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            else:
                print(f"[ERROR] API call failed: {str(e)}")
                return None

    print(f"[WARNING] Failed after {max_retries} retries")
    return None