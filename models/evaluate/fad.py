from frechet_audio_distance import FrechetAudioDistance

# to use `vggish`
frechet = FrechetAudioDistance(
    model_name="vggish",
    sample_rate=16000,
    use_pca=False, 
    use_activation=False,
    verbose=False
)
# to use `PANN`
# frechet = FrechetAudioDistance(
#     model_name="pann",
#     sample_rate=16000,
#     use_pca=False, 
#     use_activation=False,
#     verbose=False
# )
# to use `CLAP`
# frechet = FrechetAudioDistance(
#     model_name="clap",
#     sample_rate=48000,
#     submodel_name="630k-audioset",  # for CLAP only
#     verbose=False,
#     enable_fusion=False,            # for CLAP only
# )

fad_score_im4wav = frechet.score("D:/CS7643/project/ImageToAudio/test_data/ground_truth_split_s/", "D:/CS7643/project/ImageToAudio/test_data/im4wav_split_s/", dtype="float32")
fad_score_im2wav = frechet.score("D:/CS7643/project/ImageToAudio/test_data/ground_truth_split_s/", "D:/CS7643/project/ImageToAudio/test_data/im2wav_split_s/", dtype="float32")

print("im4wav: " + str(fad_score_im4wav) + "\nim2wav: " + str(fad_score_im2wav))