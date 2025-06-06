import subprocess
import requests
import time
import os
import signal
import atexit
import psutil
import sys
import json

class OllamaManager:
    """
    Manages the lifecycle of the Ollama service.
    - Checks if the Ollama service is running.
    - Starts the Ollama service if it is not running.
    - Stops the Ollama service that was started by this manager on application exit.
    """

    def __init__(self, host: str = "http://localhost:11434"):
        self.host = host
        self.ollama_process = None
        self._atexit_registered = False

    def _register_atexit(self):
        """Registers the stop method to be called at exit, if not already registered."""
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True
    
    def _unregister_atexit(self):
        """
        Unregisters the atexit handler.
        Note: Python's atexit does not support direct unregistering of a specific function.
        This is a placeholder for logic to prevent the registered function from running.
        A simple way is to set the process to None, so stop() does nothing.
        """
        # The most reliable way to prevent atexit from running is to ensure
        # the condition in stop() is false.
        self.ollama_process = None
        # We don't modify self._atexit_registered here, as it tracks registration, not execution.

    def is_ollama_running(self) -> bool:
        """Checks if the Ollama service is accessible."""
        try:
            response = requests.get(self.host)
            # Ollama root path returns "Ollama is running"
            return response.status_code == 200 and "Ollama is running" in response.text
        except requests.exceptions.ConnectionError:
            return False

    def start(self) -> str:
        """
        Starts the Ollama service if it's not already running.
        
        Returns:
            str: The status of the Ollama service: 
                 "already_running", "started", "not_found", or "failed_to_start".
        """
        if self.is_ollama_running():
            print("Ollama service is already running.")
            return "already_running"
        
        print("Ollama service not found. Attempting to start...")
        try:
            self.ollama_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print(f"Ollama service started with PID: {self.ollama_process.pid}")
            self._register_atexit() # Register cleanup only if we start the process.
            
            # Wait for the service to become available
            time.sleep(5)  # Initial wait
            
            for _ in range(10):  # Poll for up to 10 more seconds
                if self.is_ollama_running():
                    print("Ollama service has become available.")
                    return "started"
                time.sleep(1)
            
            # If it's still not running, something went wrong
            self.stop() # Clean up the zombie process
            print("Error: Ollama service was started but failed to become available.")
            return "failed_to_start"

        except FileNotFoundError:
            print("Error: 'ollama' command not found.")
            print("Please ensure Ollama is installed and the command is in your system's PATH.")
            return "not_found"
        except Exception as e:
            print(f"An unexpected error occurred while trying to start Ollama: {e}")
            return "failed_to_start"

    def stop(self):
        """Stops the Ollama service if it was started by this manager."""
        if self.ollama_process and self.ollama_process.poll() is None:
            print(f"Stopping Ollama service with PID: {self.ollama_process.pid}...")
            try:
                # Find and kill all child processes of the main process
                parent = psutil.Process(self.ollama_process.pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
                
                # Wait for the process to terminate
                self.ollama_process.wait(timeout=5)
                print("Ollama service stopped.")
            except psutil.NoSuchProcess:
                print("Ollama process already terminated.")
            except psutil.TimeoutExpired:
                print("Timeout expired while stopping Ollama. Forcing kill.")
                self.ollama_process.kill()
            except Exception as e:
                print(f"Error stopping Ollama service: {e}")
            
            # Unregister the atexit hook to prevent it from running again
            self._unregister_atexit()

    def get_installed_models(self) -> list:
        """
        Retrieves a list of locally installed Ollama models.
        Returns an empty list if the service is not running or on error.
        """
        if not self.is_ollama_running():
            print("Cannot get installed models, Ollama service is not running.")
            return []
        
        try:
            response = requests.get(f"{self.host}/api/tags")
            response.raise_for_status() # Raise an exception for bad status codes
            models_data = response.json()
            model_names = [model['name'] for model in models_data.get('models', [])]
            return model_names
        except requests.exceptions.RequestException as e:
            print(f"Failed to get installed models from Ollama: {e}")
            return []
        except json.JSONDecodeError:
            print("Failed to parse the list of installed models from Ollama.")
            return []

if __name__ == '__main__':
    # Example usage and testing of the OllamaManager
    manager = OllamaManager()
    
    # The 'atexit' registration ensures 'manager.stop()' is called on script exit.
    # To test this, you might need to run this script and then manually stop it (e.g., Ctrl+C)
    # to see the cleanup message.
    
    print("--- Ollama Manager Test ---")
    status = manager.start()
    print(f"Ollama start status: {status}")
    
    if status in ["started", "already_running"]:
        print("\n--- Checking for Installed Models ---")
        installed_models = manager.get_installed_models()
        if installed_models:
            print("Installed Ollama models:")
            for model in installed_models:
                print(f"- {model}")
        else:
            print("No installed models found or failed to retrieve them.")
    
    print("\n--- Test Script Running ---")
    print("Ollama is being managed in the background.")
    print("Script will now 'run' for 20 seconds to simulate an application.")
    print("The Ollama process will be stopped automatically on exit (if it was started by this script).")

    try:
        time.sleep(20)
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
    
    print("\n--- Test Script Finished ---")
    # The `atexit` handler will now be called, stopping the Ollama service if needed.
    # sys.exit() is called implicitly here, which triggers the atexit handlers. 