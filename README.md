# GPT_Transformer_spell_out
ChatGPT is a probabilistic system and for any one prompt, it can give us multiple answers reply to it.

We call it a language model because it models the sequence of words or characters or tokens more generally. It knows how sort of words follow each other in English language. From its perspective what it is doing is it is completing the sequence. So I give it the start of the sequence and it completes the sequence with the outcome.

This part will focus on under the hood components of what makes ChatGPT work. What's the neural net work under the hood models the sequence of these words.

The model comes from the paper called Attention is All You Need in 2017, a landmark paper in AI that proposed the Transformer architecture. GPT is short for Generatively Pre-trained Transformer.

This architecture with minor changes was copied and pasted into a huge amount of applications in AI that includes at the core of ChatGPT.

To reproduce the ChatGPT, you will train it on a huge chunk of Internet and have complicated pre-training and post-training stage. So here we will just build a Transformer based language model on character level and train it on a smaller dataset.

We are going to model how characters in Shakespeare, give the context of characters in the past to the Transformer, it will look at the characters and is going to predicted the next character that is likely to come next in the sequence.


## Reading and Exploring Data