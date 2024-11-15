"""Copyright 2024 Everlasting Systems and Solutions LLC (www.myeverlasting.net).
All Rights Reserved.

No part of this software or any of its contents may be reproduced, copied, modified or adapted, without the prior written consent of the author, unless otherwise indicated for stand-alone materials.

For permission requests, write to the publisher at the email address below:
office@myeverlasting.net

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""
from langchain_mistralai import ChatMistralAI
import getpass
import os

if "MISTRAL_KEY" not in os.environ:
    os.environ["MISTRAL_KEY"] = getpass.getpass("Enter your Mistral key: ")

llm = ChatMistralAI(
    model_name="mistral-large-latest",
    temperature=0,
    max_retries=2
)
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)



# nlp = pipeline(
#     "document-question-answering",
#     model="impira/layoutlm-document-qa"
# )
# nlp(
#     "https://templates.invoicehome.com/invoice-template-us-neat-750px.png",
#     "what is the invoice number?"
# )
# print(nlp)