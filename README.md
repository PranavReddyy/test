# Installing and Running a Chat Interface with an LLM

## 1 Installation Process

### 1.1 Install Pre-Built Chat Interface

We use **[Ollama](https://ollama.com/)**
-> Provides a simple API for creating, running, and managing models

1. Installed the package from the home page of **[Ollama](https://ollama.com/)**
2. Ran the installed application

### 1.2 Install an Open-Source Model

We use llama3.2 as our LLM model, initiated the download with :

```
ollama run llama3.2
```

#### To run it with a WebUI

We use **[Hollama](https://github.com/fmaclen/hollama)**
-> Provides a simple User Interface to interact with the installed LLM

1. Downloaded the latest supported version available from [Link](https://github.com/fmaclen/hollama/releases) which is 0.17.0 as of today.
2. Ran the installed application

## Running the interface

### Running it on the terminal

1. Initiated my terminal on macOS
2. Ran the command `ollama run llama3.2` to start interacting with the LLM
3. This is the interface I was presented with
   ![initterm](assets/infterm.png)
4. Test cases to ensure it has basic communication
   ![hterm](assets/hterm.png)
   ![trigterm](assets/trigterm.png)
   ![moneyterm](assets/moneyterm.png)

### Runninng it on the webUI

1. Open Hollama Application
2. Open the settings menu
3. Under Server section, it automatically connects with the Ollama Server locally.
4. In the available model section, selected `llama3.2:latest`
   ![setting](assets/settingh.png)
5. Started a new session in the sessions menu
   ![init](assets/inth.png)
6. Test cases to ensure it has basic communication
   ![test2.1](assets/hh.png)
   ![test2.2](assets/trigh.png)
   ![test2.3](assets/codeh.png)

