from openai import OpenAI

client = OpenAI(
    base_url="http://35.220.164.252:3888/v1/",
    # base_url="http://34.13.73.248:3888/v1", # 谷歌负载均衡网络，全球节点，适合国外访问
    # base_url="https://api.boyuerichdata.opensphereai.com/v1", # 直连香港https，数据加密
    api_key="sk-jhBYth1jQ7d3odY6nnk54Ox3AMffZyMsTCKY7Z4n4MDNZYGJ"
)

response = client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[
        {
            "role": "user",
            "content":"自我介绍一下",
        }
    ],
    temperature = 1 # 自行修改温度等参数
)

print(response)
print(response.choices[0].message.content)