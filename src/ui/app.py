import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from nicegui import ui, events
from src.utils.ollama_manager import OllamaManager

# Global state for Ollama management
ollama_manager = OllamaManager()
ollama_status: str | None = None
installed_models: list[str] = []


def setup_ollama() -> None:
    """Start or connect to the local Ollama service (runs once)."""
    global ollama_status, installed_models

    # Only attempt to start once
    if ollama_status is not None:
        return

    with ui.dialog() as dialog, ui.card():
        ui.label("Connecting to Ollama service…")
    dialog.open()

    status = ollama_manager.start()
    ollama_status = status

    if status in ("already_running", "started"):
        installed_models = ollama_manager.get_installed_models()

    dialog.close()


def render_sidebar() -> None:
    """Render the persistent sidebar (drawer) with configuration and controls."""
    with ui.drawer(side='left', value=True).classes("bg-grey-2 flex flex-col p-4 space-y-4"):
        ui.label("Configuration").classes("text-h6")

        # Ollama status feedback
        if ollama_status == "already_running":
            ui.badge("Connected to existing Ollama service.", color="green", outline=True)
        elif ollama_status == "started":
            ui.badge("Ollama service started successfully.", color="green", outline=True)
        elif ollama_status == "not_found":
            ui.badge("Ollama not found. Please install it.", color="red", outline=True)
            return  # Do not render further sidebar options
        elif ollama_status == "failed_to_start":
            ui.badge("Failed to start Ollama. Check for conflicts.", color="red", outline=True)
            return

        # Model selection dropdown
        if installed_models:
            ui.select(installed_models, label="Choose LLM Model", value=installed_models[0])
        else:
            ui.label("No Ollama models found.")

        ui.separator()
        ui.label("Service Management").classes("text-h6")

        def stop_service() -> None:
            ollama_manager.stop()
            ui.notify("Ollama service stopped.", type="positive")

        ui.button("Stop Ollama Service", on_click=stop_service)
        ui.markdown("Ollama is automatically managed. It will shut down if this app started it and you close the app.")


def render_main_page() -> None:
    """Render the main interaction page."""
    ui.label("MedLegalLLM App").classes("text-h4 mb-4")

    # -------- PDF Upload Section --------
    ui.label("1. Upload PDF").classes("text-h5")

    output_area = ui.column().classes("w-full")

    async def on_file_uploaded(e: events.UploadEventArguments):
        file = e.files[0]

        output_area.clear()
        output_area.add(ui.markdown(f"File **{file.name}** uploaded."))

        async def process_pdf() -> None:
            ui.notify(f"Processing {file.name}…", type="info")
            # TODO: hook in real document processing here
            ui.notify(f"'{file.name}' processed (placeholder).", type="positive")

        ui.button(f"Process {file.name}", on_click=process_pdf)

        # -------- Query Section --------
        ui.label("2. Query Document").classes("text-h5 mt-4")
        query_input = ui.input(label="Ask a question:")

        def perform_search():
            if not query_input.value:
                ui.notify("Please enter a query first.", type="warning")
                return
            ui.notify("Searching…", type="info")
            # TODO: replace with real query logic
            ui.markdown("This is a placeholder response.")

        ui.button("Search", on_click=perform_search)

    # Upload component must be created after defining the handler
    upload = ui.upload(
        label="Choose a PDF file",
        auto_upload=True,
        multiple=False,
        on_upload=on_file_uploaded
    ).props("accept=.pdf")


@ui.page("/")
def main_page():
    """Entry point for the NiceGUI single-page app."""
    setup_ollama()

    # Render persistent sidebar first so page layout reserves space
    render_sidebar()

    # Render main content
    with ui.column().classes("p-4 space-y-4"):
        render_main_page()


def run():
    """Run the NiceGUI application."""
    ui.run(title="MedLegalLLM App", reload=False)


if __name__ == "__main__":
    run()