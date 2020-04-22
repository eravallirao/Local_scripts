def max_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true,y_pred,from_logits=False)
	if y_true == 1 and y_pred < 0.6:
		return loss
	elif y_true == 0 and y_pred > 0.6:
		return loss
	else:
		loss = 0
		return loss