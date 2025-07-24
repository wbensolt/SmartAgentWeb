from fastapi import FastAPI
from pydantic import BaseModel
from agents.vector.retrieve_project import create_project_graph
from prometheus_fastapi_instrumentator import Instrumentator

# --- OpenTelemetry Imports ---
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# --- OTEL Configuration ---
resource = Resource(attributes={"service.name": "sma-fastapi"})
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint="http://otel-collector:4318/v1/traces")
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# --- FastAPI App ---
app = FastAPI()

# Instrumentation Prometheus
Instrumentator().instrument(app).expose(app)

# Instrumentation OpenTelemetry
FastAPIInstrumentor().instrument_app(app)
RequestsInstrumentor().instrument()

# Load the agent
graph = create_project_graph().compile()

class QueryRequest(BaseModel):
    query: str

@app.post("/feasibility/invoke")
async def invoke_endpoint(request: QueryRequest):
    result = graph.invoke({"query": request.query})
    return result
