# -*- coding: utf-8 -*-


class TagsLabel():
    def __init__(self):


        self.tokenize_tag = ['O', 'B', 'I']
        self.tokenize_tag_to_idx = self.generate_tags_dict(self.tokenize_tag)
        self.tokenize_tag_size = len(self.tokenize_tag)

        self.entity_tag = ['O'] + [str(i) for i in range(1,55) if i not in (27,45)]
        self.entity_tag_to_idx = self.generate_tags_dict(self.entity_tag)
        self.entity_tag_size = len(self.entity_tag)

        self.ensemble_tag = ['O']
        for i in range(1,55):
            if i not in (27,45):
                for t in self.tokenize_tag[1:]:
                    self.ensemble_tag.append(t+'-'+str(i))
        self.ensemble_tag_to_idx = self.generate_tags_dict(self.ensemble_tag)

    def generate_tags_dict(self, tags_list):
        return {tag:i for i,tag in enumerate(tags_list)}

    def get_item_size(self):
        return self.tokenize_tag_size, self.entity_tag_size

    def convert_item_to_id(self, item, item_dict):
        if item in item_dict:
            return item_dict[item]
        # elif self.is_word:
        #     unk = "<unk>" + str(len(item))
        #     if unk in self.item2idx:
        #         return self.item2idx[unk]
        #     else:
        #         return self.item2idx['<unk>']
        else:
            print("Label does not exist!!!!")
            print(item)
            raise KeyError()

    def convert_item_to_ids(self, items):
        tokenize_label_ids = []
        entity_label_ids = []
        for item in items:
            split_item = item.strip().split('-')
            if len(split_item) != 2:
                tokenize_item = 'O'
                entity_item = 'O'
            else:
                tokenize_item = split_item[0]
                entity_item = split_item[1]
            tokenize_label_ids.append(self.convert_item_to_id(item=tokenize_item, item_dict=self.tokenize_tag_to_idx))
            entity_label_ids.append(self.convert_item_to_id(item=entity_item, item_dict=self.entity_tag_to_idx))
        return tokenize_label_ids, entity_label_ids


    def convert_id_to_item(self, tokenize_id, entity_id):
        if tokenize_id >= 0 and tokenize_id < self.tokenize_tag_size \
                and entity_id >= 0 and entity_id < self.entity_tag_size:
            # print("debug1 ",tokenize_id,entity_id)
            if self.tokenize_tag[tokenize_id] == 'O' or self.entity_tag[entity_id] == 'O':
                # print("debug2 ",tokenize_id,entity_id)
                return 'O'
            else:
                return self.tokenize_tag[tokenize_id] + '-' + self.entity_tag[entity_id]
        else:
            # print("debug3 ",tokenize_id,entity_id)
            return 'O'

    def get_ensemble_id(self, tokenize_id, entity_id):
        if tokenize_id >= 0 and tokenize_id < self.tokenize_tag_size \
                and tokenize_id >= 0 and tokenize_id < self.tokenize_tag_size:
            if self.tokenize_tag[tokenize_id] == 'O' or self.entity_tag[entity_id] == 'O':
                return self.ensemble_tag_to_idx['O']
            else:
                return self.ensemble_tag_to_idx[self.tokenize_tag[tokenize_id] + '-' + self.entity_tag[entity_id]]
        else:
            return self.ensemble_tag_to_idx['O']

    def get_ensemble_ids(self, tokenize_ids, entity_ids):
        return [self.get_ensemble_id(t, e) for t,e in zip(tokenize_ids, entity_ids)]





