from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableLambda,RunnableParallel

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-2.0-flash")

summary_template = ChatPromptTemplate.from_messages([("system","You are a movie critic"),
                                                     ("human","Give me a brief summary of this movie {movie_name}.")])

def plot_analysis (plot) : 
    plot_template =  ChatPromptTemplate.from_messages([("system","You are a movie critic"),
                                                    ("human","Analyze the plot: {plot}. What are its strengths and weaknesses?")])
    return plot_template.invoke(plot)
def char_analysis (plot) :
    char_template =  ChatPromptTemplate.from_messages([("system","You are a movie critic"),
                                                   ("human","Analyze the characters from this plot: {plot}. What are their strengths and weaknesses?")])
    return char_template.invoke(plot)

def combine(plot,char) : 
    return f"Plot analysis : \n {plot} \n\n\n\n Character Analysis : \n {char}"

plot_chain =  ( RunnableLambda(lambda plot : plot_analysis(plot)) | llm | StrOutputParser() )
char_chain = (RunnableLambda(lambda plot : char_analysis(plot)) | llm | StrOutputParser() )

chain = ( 
         summary_template | llm | StrOutputParser() 
         | RunnableLambda(lambda summary : {"plot" : summary})
         | RunnableParallel(branches = {"plot" : plot_chain, "characters" : char_chain} ) 
         | RunnableLambda(lambda x : combine(x["branches"]["plot"],x["branches"]["characters"]))
)

result = chain.invoke({"movie_name" : "Interstellar"})
print(result)
