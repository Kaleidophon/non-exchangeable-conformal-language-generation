"""
Perform hallucinations experiment, where we compare we first let the model generate freely, and then feed the same tokens
back into the decoder, but restricting the attention on the source side. We then compare the two set sizes.
"""