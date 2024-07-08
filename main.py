import os

os.environ["LANGCHAIN_PROJECT"] = "LangGraph Tutorial"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["OPENAI_API_KEY"] = "sk-proj-#"
os.environ["LANGCHAIN_API_KEY"] = "lsv2_pt_#"

from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import CommaSeparatedListOutputParser


def create_todo_list(goal):
    output_parser = CommaSeparatedListOutputParser()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 나의 할일을 관리하는 비서입니다. 당신의 임무는 나를 위하여 주어진 목표에 대하여 할일 목록을 작성하는 것입니다.",
            ),
            (
                "human",
                "주어진 목표(goal)를 잘 수행하기 위하여 할일 목록을 작성해 주세요. `할일:걸리는시간(hour)` 으로 작성하세요. 걸리는 시간은 반올림하여 int 로 작성하세요.\n\n#목표: {goal}\n\n#형식: {format_instuctions}",
            ),
        ]
    )
    prompt = prompt.partial(format_instuctions=output_parser.get_format_instructions())
    llm = ChatOpenAI(model_name="gpt-4-turbo")
    chain = prompt | llm | CommaSeparatedListOutputParser()

    output = chain.invoke({"goal": goal})
    print(output)
    return output


from typing import TypedDict


class GraphState(TypedDict):
    goal: str  # 목표
    todo: list[str]  # 할 일 목록
    current_job: str  # 현재 작업
    total_time: int  # 총 소요시간(시간)
    time_spent: int  # 소요 시간(시간)
    status: str  # 상태(진행중, 다음 작업, 종료)


# update GrapState.todo with making goal with create_todo_list function and update graph goal
def list_todo(state: GraphState) -> GraphState:
    goal = state["goal"]
    todo = create_todo_list(goal)
    state["todo"] = todo
    return state


def start_job(state: GraphState) -> GraphState:
    todo = state["todo"]
    if len(todo):
        current_job, total_time = todo.pop(0)
        status = "진행중"
        time_spent = 0
    return GraphState(
        current_job=current_job,
        total_time=total_time,
        status=status,
        time_spent=time_spent,
    )

def process_job(state: GraphState) -> GraphState:
    time_spent = state["time_spent"]
    time_spent += 1

    return GraphState(time_spent=time_spent)


def check_progress(state: GraphState) -> GraphState:
    if state["time_spent"] >= state["total_time"]:
        status = "다음 작업"
        if len(state["todo"]) == 0:
            status = "종료"
    else:
        status = "진행중"
    return GraphState(status=status)


def next_step(state: GraphState) -> GraphState:
    return state["status"]


from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from IPython.display import Image, display
from PIL import Image
import io


if __name__ == '__main__':
    # langgraph.graph에서 StateGraph와 END를 가져옵니다.
    workflow = StateGraph(GraphState)

    # Todo 를 작성합니다.
    workflow.add_node("list_todo", list_todo)  # 에이전트 노드를 추가합니다.

    # Todo 작업을 시작합니다.
    workflow.add_node("start_job", start_job)

    # 작업을 진행합니다.
    workflow.add_node("process_job", process_job)

    # 작업을 중간 체크합니다.
    workflow.add_node("check_progress", check_progress)

    # 각 노드들을 연결합니다.
    workflow.add_edge("list_todo", "start_job")
    workflow.add_edge("start_job", "process_job")
    workflow.add_edge("process_job", "check_progress")

    # 조건부 엣지를 추가합니다.
    workflow.add_conditional_edges(
        "check_progress",  # 관련성 체크 노드에서 나온 결과를 is_relevant 함수에 전달합니다.
        next_step,
        {
            "진행중": "process_job",  # 관련성이 있으면 종료합니다.
            "다음 작업": "start_job",  # 관련성이 없으면 다시 답변을 생성합니다.
            "종료": END,  # 관련성 체크 결과가 모호하다면 다시 답변을 생성합니다.
        },
    )

    # 시작점을 설정합니다.
    workflow.set_entry_point("list_todo")

    # 기록을 위한 메모리 저장소를 설정합니다.
    memory = MemorySaver()

    # 그래프를 컴파일합니다.
    app = workflow.compile(checkpointer=memory)

    try:
        # 실행 가능한 객체의 그래프를 mermaid 형식의 PNG로 그려서 이미지 데이터를 가져옵니다.
        image_data = app.get_graph(xray=True).draw_mermaid_png()

        # 이미지 데이터를 파일로 저장
        with open('graph.png', 'wb') as f:
            f.write(image_data)

        # 이미지 파일 불러와서 표시
        image = Image.open('graph.png')
        image.show()
    except Exception as e:
        print(f"An error occurred: {e}")
        # 이 부분은 추가적인 의존성이 필요하며 선택적으로 실행됩니다.
        pass