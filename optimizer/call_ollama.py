import requests
import json
from optimizer.config import FACTOR_OPTIMIZATION_CONFIG

def check_ollama_status():
    """检查Ollama服务是否运行"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def list_available_models():
    """列出可用的Ollama模型"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except:
        return []

def call_ollama(sys_message, hum_message):
    """
    使用Ollama Chat API调用本地模型
    
    参数:
        sys_message: 系统消息（设定角色和行为规范），可以为None
        hum_message: 用户消息（具体的提问内容）
        model: 要使用的模型名称，默认为"gemma:2b"
        temperature: 生成温度，值越高结果越随机，默认为0.7
    
    返回:
        模型生成的文本
    """
    # Ollama Chat API端点
    url = FACTOR_OPTIMIZATION_CONFIG['OLLAMA']['BASE_URL']
    
    # 构建消息数组
    messages = []
    if sys_message:
        messages.append({
            "role": "system",
            "content": sys_message
        })
    
    messages.append({
        "role": "user", 
        "content": hum_message
    })
    
    # 请求数据
    data = {
        "model": FACTOR_OPTIMIZATION_CONFIG['OLLAMA']['MODEL_NAME'],
        "messages": messages,
        "temperature": FACTOR_OPTIMIZATION_CONFIG['OLLAMA']['TEMPERATURE'],
        "stream": False  # 非流式返回，一次性获取完整结果
    }
    
    try:
        # 发送POST请求
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(data)
        )
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            message = result.get("message", {})
            return message.get("content", "")
        else:
            return f"请求失败，状态码: {response.status_code}"
    
    except Exception as e:
        return f"调用出错: {str(e)}"


if __name__ == "__main__":
    print("🚀 Ollama模型调用测试")
    print("=" * 60)
    
    # 检查Ollama服务状态
    print("🔍 检查Ollama服务状态...")
    if not check_ollama_status():
        print("❌ Ollama服务不可用")
        print("请确保Ollama已启动: http://localhost:11434")
        exit(1)
    else:
        print("✅ Ollama服务正常运行")
    
    # 列出可用模型
    available_models = list_available_models()
    if available_models:
        print(f"📋 可用模型 ({len(available_models)} 个):")
        for i, model in enumerate(available_models, 1):
            print(f"   {i}. {model}")
    else:
        print("⚠️ 没有可用模型")
        exit(1)
    
    # 选择要测试的模型
    model = available_models[0]  # 使用第一个可用模型
    print(f"\n🎯 使用模型: {model}")
    
    # 测试参数
    sys_message = "你是一个专业的AI助手，请用简洁明了的方式回答问题。"
    hum_message = "请简要介绍一下人工智能的发展历程"
    temperature = 0.7
    
    print(f"\n📝 测试参数:")
    print(f"   系统消息: {sys_message}")
    print(f"   用户提问: {hum_message}")
    print(f"   温度参数: {temperature}")
    
    print("\n🔧 方法1: 使用Generate API")
    print("-" * 40)
    try:
        response1 = call_ollama(sys_message, hum_message)
        print("回答:", response1[:200] + "..." if len(response1) > 200 else response1)
    except Exception as e:
        print(f"❌ Generate API调用失败: {e}")
    
    print("\n✅ 测试完成!")
