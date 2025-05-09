import subprocess
import requests
import time
import os
import signal
import atexit

class OllamaManager:
    def __init__(self):
        self.process = None
        atexit.register(self.stop)
    
    def start(self) -> str:
        """Start Ollama service and return its status.
        Returns:
            str: "already_running", "started", "not_found", "failed_to_start"
        """
        try:
            # Check if Ollama is already running with a timeout
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                return "already_running"
        except requests.exceptions.RequestException: # Catches connection errors, timeouts, etc.
            pass # If it's not running or not responding, we'll try to start it

        try:
            # Start Ollama process, redirecting stdout/stderr to prevent pipe buffer issues
            self.process = subprocess.Popen(
                ['ollama', 'serve'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Wait for Ollama to start (max 10 seconds)
            for _ in range(10):
                time.sleep(1) # Give it a second to initialize
                try:
                    response = requests.get('http://localhost:11434/api/tags', timeout=1)
                    if response.status_code == 200:
                        return "started"
                except requests.exceptions.RequestException:
                    continue # Keep trying until timeout
            
            # If loop finishes, Ollama didn't become responsive
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2) # Wait a bit for termination
                except subprocess.TimeoutExpired:
                    self.process.kill() # Force kill if terminate fails
                self.process = None
            return "failed_to_start"

        except FileNotFoundError:
            self.process = None # Ensure process is None
            return "not_found"
        except Exception: # Catch other potential errors during Popen (e.g., permissions)
            self.process = None # Ensure process is None
            return "failed_to_start" # Generic failure
    
    def stop(self) -> None:
        """Stop Ollama service if it was started by this manager."""
        if self.process:
            # print("Attempting to stop Ollama process...") # Optional: for debugging
            try:
                os.kill(self.process.pid, signal.SIGTERM) # Send Terminate signal
                self.process.wait(timeout=5) # Wait for graceful shutdown
                # print("Ollama process terminated gracefully.") # Optional
            except ProcessLookupError:
                pass # Process already died
            except subprocess.TimeoutExpired:
                # print("Ollama process did not terminate gracefully, sending SIGKILL...") # Optional
                try:
                    os.kill(self.process.pid, signal.SIGKILL) # Force kill
                    # print("Ollama process killed.") # Optional
                except ProcessLookupError:
                    pass # Process already died
                except Exception: # Other errors during kill
                    # print(f"Error trying to kill Ollama process: {e}") # Optional
                    pass
            except Exception: # Other errors during terminate/wait
                # print(f"Error stopping Ollama process: {e}") # Optional
                pass 
            finally:
                self.process = None
    
    def get_installed_models(self) -> list:
        """Get list of models installed in Ollama."""
        try:
            response = requests.get('http://localhost:11434/api/tags', timeout=2)
            if response.status_code == 200:
                models_data = response.json().get('models', [])
                return [model['name'] for model in models_data]
            return []
        except requests.exceptions.RequestException:
            return [] 