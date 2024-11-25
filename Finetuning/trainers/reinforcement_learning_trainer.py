from textrl import TextRLActor, train_agent_with_evaluation
from trl.trainer.dpo_config import DPOConfig
from trl.trainer.dpo_trainer import DPOTrainer
from Finetuning.support import support
from Finetuning.support.reinforcement_learning_environment import ReinforcementLearningEnvironment
from Finetuning.support.support import MAX_SEQ_LEN
from datasets import Dataset


def reinforcement_learning_train_model(algorithm, model, tokenizer, observation_list, epochs, steps, models_folder):
    model.train()
    if algorithm == "ppo":
        environment = ReinforcementLearningEnvironment(model, tokenizer, observation_input=observation_list, max_length=MAX_SEQ_LEN, compare_sample=50)
        actor = TextRLActor(environment, model, tokenizer, act_deterministically=False, temperature=0.8, top_k=5, top_p=1.0)
        agent = actor.agent_ppo(update_interval=10, minibatch_size=50, epochs=epochs)
        train_agent_with_evaluation(agent, environment, steps=steps, eval_n_steps=None, eval_n_episodes=80, eval_interval=20, outdir=models_folder)

    elif algorithm == "dpo":
        # LoRA configuration
        training_args = DPOConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=32,
            gradient_checkpointing=True,
            learning_rate=5e-5,
            lr_scheduler_type="cosine",
            max_steps=steps,
            logging_steps=10,
            output_dir=models_folder,
            optim="paged_adamw_32bit",
            warmup_steps=100,
            report_to="wandb",
        )
        # Create DPO trainer
        dpo_trainer = DPOTrainer(
            model,
            args=training_args,
            train_dataset=Dataset.from_list(observation_list),
            tokenizer=tokenizer,
            #peft_config=adapter,
            beta=0.8,
        )
        # Fine-tune model with DPO
        dpo_trainer.train()

    else:
        raise Exception("Unknown reinforcement learning algorithm!")

    # Saving model
    support.save_model(model, tokenizer, "reinforced")
