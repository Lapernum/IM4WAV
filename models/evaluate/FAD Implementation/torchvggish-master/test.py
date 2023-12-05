import FAD_evaluation

audio1 = FAD_evaluation.evaluation_model_forward("bus_chatter.wav")
audio2 = FAD_evaluation.evaluation_model_forward("bus_chatter.wav")
print(FAD_evaluation.calculate_frechet_distance(audio1, audio2))