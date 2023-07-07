from langchain.chains import LLMChain
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory


class MemoryPromptTemplate:
    def __init__(self, llm, input_variables, template):
        self.llm = llm

        self.input_variables = input_variables
        self.template = template
        self.prompt = PromptTemplate(input_variables=self.input_variables, template=self.template)
        self.memory = ConversationBufferMemory(memory_key="chat_history", input_key="query")
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)
    
    def process(self, input_data):
        return self.chain.run(input_data)

class BasePromptTemplate:
    def __init__(self, llm, input_variables, template):
        self.llm = llm

        self.input_variables = input_variables
        self.template = template
        self.prompt = PromptTemplate(input_variables=self.input_variables, template=self.template)
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def process(self, input_data):
        return self.chain.run(input_data)

class QuestionAnswerTemplate(MemoryPromptTemplate):
    def __init__(self, llm):
        self.input_variables = ['query', 'sources', 'chat_history']
        self.template_string = """You are a chatbot that helps answer questions based on text.

        Given the following passages from a document and the user question at the bottom, create a final answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Organize your answer into logical chunks as needed and be as informative, clear and concise as possible. Return an answer is 1-3 sentences long.

        Context: ```
        {sources}
        ```

        {chat_history}
        Human:{query}
        AI:
        """
        super().__init__(llm, self.input_variables, self.template_string)

class ClassifierTemplate(BasePromptTemplate):
    def __init__(self, llm):
        self.input_variables = ['text', 'classes']
        self.template_string = """You are a classification expert. Classify the text below into one of the following:

        Class choices: [{classes}]

        Text to classify: {text}

        Classification:
        """
        super().__init__(llm, self.input_variables, self.template_string)


class DraftResponseTemplate(BasePromptTemplate):
    def __init__(self, llm):
        self.input_variables = ['message', 'context']
        self.template_string = """You are an expert in communication. Based on the given context, draft a suitable response to the message below:

        Context: {context}

        Message received: {message}

        Drafted response:
        """
        super().__init__(llm, self.input_variables, self.template_string)

class NERExtractionTemplate(BasePromptTemplate):
    def __init__(self, llm):
        self.input_variables = ['text']
        self.template_string = """You are an expert in named entity recognition (NER). Analyze the following text and identify any entities that fall into categories such as 'person', 'organization', 'location', 'date', and so on:

        Text: {text}

        Entities:
        """
        super().__init__(llm, self.input_variables, self.template_string)

class DomainDocumentSummarizationTemplate(BasePromptTemplate):
    def __init__(self, llm):
        self.input_variables = ['text', 'domain', 'summary_length']
        self.template_string = """
        Summarize the following {domain} domain documents in about {summary_length} words.

        Document 1: "The cardiovascular system consists of the heart, blood vessels, and the approximately 5 liters of blood that the blood vessels transport. Responsible for transporting oxygen, nutrients, hormones, and cellular waste products throughout the body, the cardiovascular system is powered by the body’s hardest-working organ — the heart."
        Summary 1: { "Document": "The cardiovascular system consists of...", "Summary": "The cardiovascular system, composed of the heart, blood vessels, and about 5 liters of blood, transports oxygen, nutrients, hormones, and cellular waste throughout the body. The heart, the body's most hardworking organ, powers the system." }

        Document 2: "AI refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving."
        Summary 2: { "Document": "AI refers to the simulation of...", "Summary": "AI is the replication of human intelligence in machines, enabling them to think and act like humans. Machines that demonstrate human-like characteristics, such as learning and problem-solving, can also be considered AI." }

        Document 3: {text}

        Summary 3: {}
        """
        super().__init__(llm, self.input_variables, self.template_string)


class CustomerResponseSummarizationTemplate(BasePromptTemplate):
    def __init__(self, llm):
        self.input_variables = ['text', 'summary_length']
        self.template_string = """
        Summarize the following customer responses or email chains in 150 words or less..

        Text 1: "I've been using your product for a month and I've noticed a significant improvement in my workflow. However, I have been experiencing some minor bugs when trying to use the export function. Could you please look into this? Its really frustrating. Everything else is great though. Yeah thanks."
        Summary 1: { "OriginalText": "I've been using your product...", "Summary": "Customer has noticed an improvement in workflow with our product, but is experiencing minor bugs with the export function." }

        Text 2: "Hello, this is Jim Callahoun and I'd like to cancel my subscription to your service. I found another service that fits my needs better. I would like to do that as soon as possible. Feel free to call back or just cancel. I don't really care. Thank you for your support during this time."
        Summary 2: { "OriginalText": "Hello, I'd like to cancel...", "Summary": "Customer Jim Callahoun wants to cancel their subscription as they found another service that better suits their needs." }

        Text 3: {text}

        Summary 3: {}
        """
        super().__init__(llm, self.input_variables, self.template_string)

class NERExtractionTemplate(BasePromptTemplate):
    def __init__(self, llm):
        self.input_variables = ['text']
        self.template_string = """
        Extract entities from the corresponding texts below.

        Text 1: " Hi, my name is John Hayes. J O H N H A Y E S and my email is accounting AC C O UN T I N G at D as in David, F as in Franklin, M as in Michael Tole, T 00 L W O R K S. So the full address is accounting at D F M tool works dot com. And I have a question regarding a 10 99 K issue regarding the gross amount paid on the or what was stated on the statement. And I was hoping to speak to a sales or a service representative regarding this issue. If someone please call me back, my number is 9306007023 or you can also reach me at my email. I'll repeat that again. It's accounting at D F M tool works dot com. And I have a question regarding the gross amount stated on that statement for 2022. If someone please contact me at their earliest convenience, that'd be greatly appreciated. Thank you. Bye."
        Entities 1: { "Person": ["John Hayes"], "Email": ["accounting@dfmtoolworks.com"], "Phone": ["9306007023"] }

        Text 2: "This is Abdi Ghoulam Ahramadaoui, I received on my Stranger Mobl application the 1099 tax form, but unfortunately I cant fill my tax yet because I was working with Stranger Mobl in TWO different states California and Nevada and I received only one 1099 form from California. I tried to change my basic information( my address) on my application but I couldnt . Im asking you if you can send me again two different 1099 for each state. I worked in California from 03/01/2022 to 09/30/2022 ( address is 9363 Civic Blvd Unit 2A concord CA 94520) and I worked in Nevada from 10/01/2022 ( address is South Soho Ave apt 19 Las Vegas NV 88901) And I would pleasure that if you change my address as well because Im always trying to change it but it doesnt let me do it. Hopefully you understand me and thank you. Regards Abdi"
        Entities 2: { "Person": ["John Hayes"], "Email": ["accounting@dfmtoolworks.com"], "Phone": ["9306007023"] }

        Text 3: {text}

        Entities 3: {} 
        """
        super().__init__(llm, self.input_variables, self.template_string)

