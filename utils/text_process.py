import numpy as np

def process_file(retrieved_text_path):

    loaded_data = np.load(retrieved_text_path, allow_pickle=True)

    data_dict = {}
    for item in loaded_data:
        key = item['image_id']
        tags = [tag.replace('a photo of a ', '').strip() for tag in item['tags']]
        attributes = [attribute.strip() for attribute in item['attributes']]
        sub_dict = {
            'tags': tags,
            'attributes': attributes 
        }
        data_dict[key] = sub_dict
    return data_dict