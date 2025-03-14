# Environment Setup Guide for Last-Mile Delivery Platform

## Prerequisites

Ensure you have the following installed:

- **Operating System**: Windows 10/11, macOS, or Linux
- **.NET SDK**: .NET 9.0 (latest stable version)
- **Database**: SQLite
- **Frontend Framework**: React-Admin
- **IDE**: Visual Studio 2022 (latest version) or JetBrains Rider / VS Code
- **Git**: Version control system
- **Postman**: API testing tool
- **Node.js & npm**: Required for frontend development
- **Python 3.x**: Required for algorithm module
- **Virtual Environment**: `venv` or `conda` for Python dependency management

## Backend Setup (API)

### 1. Download the Backend Repository

```sh
git clone git@github.com:tianwen-zhou/LogisticsApi.git
cd LogisticsApi
```

### 2. Install .NET 9.0 SDK

Download and install the latest .NET 9.0 SDK from:
[.NET SDK Download](https://dotnet.microsoft.com/en-us/download)

Verify installation:

```sh
dotnet --version
```

### 3. Set Up Development Environment

#### Windows

- Install **Visual Studio 2022** with the following workloads:
  - **ASP.NET and web development**
  - **.NET Core cross-platform development**

#### macOS & Linux

- Install **Visual Studio Code** or **JetBrains Rider**
- Install C# extension in VS Code (if applicable)

### 4. Setup Database (SQLite)

```sh
dotnet add package Microsoft.EntityFrameworkCore.Sqlite
```

Update `appsettings.json`:

```json
"ConnectionStrings": {
  "DefaultConnection": "Data Source=logistics.db"
}
```

Run migrations:

```sh
dotnet ef migrations add InitialCreate
dotnet ef database update
```

### 5. Run the API

```sh
dotnet run
```

### 6. Test the API

Use Postman or curl to test the API endpoint:

```sh
curl http://localhost:5137/api/Drivers/1
```

### 7. Setup Version Control

```sh
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Frontend Setup (React-Admin)

### 1. Download the Frontend Repository

```sh
git clone https://github.com/tianwen-zhou/LogisticsFrontend.git
cd LogisticsFrontend
```

### 2. Install Node.js

Download and install the latest version of Node.js from:
[Node.js Download](https://nodejs.org/)

Verify installation:

```sh
node -v
npm -v
```

### 3. Install Dependencies

```sh
npm install
```

### 4. Configure Backend API

Update the API base URL in the project configuration:

```json
API_URL = "http://localhost:5137/api"
```

### 5. Start the Development Server

```sh
npm run dev
```

The frontend will be accessible at:
[http://localhost:5173/](http://localhost:5173/)

### 6. Build for Production

```sh
npm run build
```

### 7. Run Tests

```sh
npm run test
# or
yarn run test
```

### 8. Data Provider

The included data provider uses **FakeREST** to simulate a backend. The `src/data.json` file contains test data with the following resources:

- **Posts**: `id`, `title`, `content`
- **Comments**: `id`, `post_id`, `content`

### 9. Authentication

The included auth provider is for development/testing purposes. The `src/users.json` file contains test users:

- **janedoe / password**
- **johndoe / password**

## Algorithm Module Setup (Python)

### 1. Download the Algorithm Repository

```sh
git clone https://github.com/tianwen-zhou/vrp-solution.git
cd vrp-solution
```

### 2. Set Up Python Virtual Environment

```sh
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

The `requirements.txt` file includes essential packages such as `pyvrp` and `OR-tools`.

### 4. Load the Lade-D Dataset in Python

The experimental dataset is available at:
[LaDe-D Dataset](https://huggingface.co/datasets/Cainiao-AI/LaDe-D)

Load it directly in your Python script:

```python
from datasets import load_dataset

# Load LaDe-D dataset
ds = load_dataset("Cainiao-AI/LaDe-D")
data_jl = ds['delivery_jl']
```

### 5. Run an Example Script

Run specific algorithm implementations from the `vrp-solution/v1` directory:

#### OR-Tools Algorithm
```sh
python vrp-solution/v1/or-stat.py
```

#### ACO Algorithm
```sh
python vrp-solution/v1/aco-stat.py
```

#### RL Algorithm
```sh
python vrp-solution/v1/rl-stat.py
```

#### Hybrid ACO + RL Algorithm
```sh
python vrp-solution/v1/hybrid-rl-aco.py
```

## Additional Configurations

- **CORS Setup**: Modify `Program.cs` to allow cross-origin requests
- **Logging**: Configure logging in `appsettings.json`
- **Authentication**: Use JWT or OAuth2 for API security
- **Containerization**: Use Docker for deployment

This completes the setup for the last-mile delivery platform, including the API, frontend, and algorithm module.
