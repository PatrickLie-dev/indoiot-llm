from huggingface_hub import HfApi
api = HfApi()

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'demo.launch(\n        server_name="0.0.0.0",\n        server_port=port,\n        share=False,\n    )',
    'demo.launch()'
)

api.upload_file(
    path_or_fileobj=content.encode('utf-8'),
    path_in_repo='app.py',
    repo_id='Pat-L/indoiot-llm',
    repo_type='space'
)
print('app.py updated!')
