Simple FAQs Chatbot

This is a simple chatbot created using TensorFlow and Python, By the use of deep learning to automate conversational responses. My goal was to build a model that understands and generates relevant replies to user inputs based on a predefined set of conversational pairs.
I tackled the challenge of developing an intelligent chatbot that can provide contextually appropriate responses. Traditional rule-based chatbots often struggle to handle a wide range of inputs effectively. By leveraging deep learning, I aimed to enhance the chatbot's ability to understand and respond to user’s queries more accurately and naturally.
I used TensorFlow, which is a widely-used deep learning library, along with Python to develop the chatbot. Other essential tools that included are `LabelEncoder` from the ‘ sklearn.preprocessing ’ module used for label encoding and the ‘ tf.keras.preprocessing.text.Tokenizer ‘ for text tokenisation.
1. Data Preparation :
				 I collected conversational pairs from GitHub repository and split them into inputs and outputs. These pairs then formed the foundation for training the model.
2. Label Encoding : 
				The ‘ LabelEncoder ‘ was employed to convert the output responses into numerical labels, thus making them suitable for the model.
3. Tokenization : 
				‘ Tokenizer ‘ is used to tokenise and convert the input texts into sequences.
4. Model Building : 
				A sequential neural network model was constructed using TensorFlow. The architecture included an embedding layer for text representation, an LSTM layer for 				handling sequential data, and a dense layer with a softmax activation function for classification.
5. Training : 
				Then the model was trained on the prepared sequences and labels over 100 epochs to learn the patterns in the data.

Design of the Project :
1. Data Collection : Gathered conversational pairs from GitHub repository.
2. Data Preprocessing : Encoded the labels and tokenised the inputs.
3. Model Construction : Built and compiled the neural network model.
4. Model Training : Trained the model using the prepared data.
5. Response Generation : Used the trained model to generate responses to new user inputs.

Conclusion :
					The outcome of this project is a functional chatbot that is capable of delivering accurate and contextually appropriate responses based on predefined conversational pairs. This project can be used demonstrates the potential of deep learning techniques to enhance chatbot capabilities. This approach allows the chatbot to understand and respond to a wide variety of user inputs with improved accuracy and relevance compared to traditional rule-based systems.
