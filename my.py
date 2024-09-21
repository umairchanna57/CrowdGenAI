from transformers import BlipProcessor, BlipForConditionalGeneration, BlipConfig

# Load the configuration for the model
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Ensure this is a valid model name in the Hugging Face Hub

# Load the configuration of the model
config = BlipConfig.from_pretrained(model_name)

# Modify embed_dim to a value divisible by num_heads (3600 in this case)
config.vision_config.embed_dim = 3600  # Adjusted to be divisible by 12
config.vision_conzzzfig.num_heads = 12    # Keeping the number of heads as 12

# Load the model using the updated configuration
model = BlipForConditionalGeneration(config)

# Load the processor
processor = BlipProcessor.from_pretrained(model_name)

# Save the updated model and processor
processor.save_pretrained("./Llama")
model.save_pretrained("./Llama")

print("Model and processor saved successfully!")
