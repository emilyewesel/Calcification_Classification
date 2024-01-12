import torch
from torch import nn
from pytorch_lightning.core.module import LightningModule
from torch.nn import functional as F
from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
from settings import IMAGE_SIZE, FEATURES, BATCH_SIZE, TARGET
import torchmetrics
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
import matplotlib.pyplot as plt
import shap
from google.cloud import translate_v2 as translate
import random
import nltk
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('punkt')
import matplotlib.pyplot as plt
import torchio as tio



import os
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MultiModModelWithLanguage(LightningModule):
    '''
    Resnet Model Class including the training, validation and testing steps
    '''

    def __init__(self, class_weight, scaler):

        super().__init__()

        self.class_weight = class_weight

        self.scaler = scaler
        self.do_transform = True

        self.resnet = resnet10(pretrained=False,
                               spatial_dims=3,
                               num_classes=120, # might change this one
                               n_input_channels=1
                               )
        # base = 'michiyasunaga/BioLinkBERT-base'
        # base = "roberta-base"
        # base = "distil-bert-base-uncased"
        base = "bert-base-uncased"
        # base = "distilroberta-base"

        # qa_pipeline = pipeline("question-answering", model="medalpaca/medalpaca-7b", tokenizer="medalpaca/medalpaca-7b")
        # question = "What are the symptoms of diabetes?"
        # context = "Diabetes is a metabolic disease that causes high blood sugar. The symptoms include increased thirst, frequent urination, and unexplained weight loss."
        # answer = qa_pipeline({"question": question, "context": context})
        self.tokenizer = AutoTokenizer.from_pretrained(base, cache_dir = "/scratch/users/ewesel/")
        self.language_model = AutoModel.from_pretrained(base, cache_dir = "/scratch/users/ewesel/")
                                                
        # Freeze weights so those don't get trained        
        for param in self.language_model.parameters():
            param.requires_grad = False


        self.NUM_FEATURES = len(FEATURES)

        # self.NUM_FEATURES = 0

        # fc layer to make image size same as tabular data size
        # self.fc = nn.Linear(400, 1)

        # combine resnet with final fc layer
        # self.imagenet = nn.Sequential(self.resnet, self.fc)
        # fc layer that maps language model inputs to smaller dimension
        self.language_fc = nn.Linear(768, 120)

        # fc layer for tabular data. We substract 31 because age and sex are encoded as sentences
        self.fc1 = nn.Linear((self.NUM_FEATURES-0), 120)

        # first fc layer which takes concatenated input
        self.fc2 = nn.Linear((240 + 0 + 0), 32)

        # final fc layer which takes concatenated imput
        self.fc3 = nn.Linear(32, 1)

        self.train_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.val_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.test_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)

        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)

        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)
        self.train_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

        self.train_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

        self.val_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

        self.val_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

        self.test_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')
        
        self.test_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

        self.results_column_names = ['subject', 'label', 'prediction', 'age', 'sex']

        self.train_results_df = pd.DataFrame(columns=self.results_column_names)

        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.val_results_df = pd.DataFrame(columns=self.results_column_names)
    # def forward(self, img, tab):
    #     """
    #     x is the input data
    #     """
    #     # run the model for the image
        
    #     # To run the correlation triplet
    #     # print(img.shape)
    #     img = torch.unsqueeze(img, 1)
    #     img = img.to(torch.float32)
    #     # print(img.shape)
    #     img = self.resnet(img)
    #     # change the dtype of the tabular data
    #     tab = tab.to(torch.float32)
    #     tab = F.relu(self.fc1(tab))
    #     tab = F.relu(self.fc2(tab))
    #     img = F.relu(self.fc2(img))
    #     img = img.unsqueeze(0)
    #     tab = tab.unsqueeze(1)
    #     x = F.conv1d(img, tab, padding=self.embedding_dimension - 1, groups=img.size(1))
    #     x = x.squeeze()
    #     # get the final concatenated embedding
    #     out1 = self.fc7(x)
    #     out2 = self.fc5(F.relu(out1))
    #     out2 = torch.squeeze(out2)
    #     return out2, out1

    def forward(self, img, tab=None):
        """

        x is the input data

        """
        # run the model for the image
        self.language_model = self.language_model.to('cuda')
        self.tokenizer = self.tokenizer
        
        # print(img.shape)

       
        img = torch.unsqueeze(img, 1)
        img = img.to(torch.float32)

        img = self.resnet(img)
        # torch.set_printoptions(threshold=float('inf'))
        # print(img[0, 0, :, :])


        # print("imageee", img.shape)


        # import matplotlib.pyplot as plt

        # # Assuming feature_maps shape is (batch_size, channels, height, width)
        # # You can select a specific example from the batch if needed
        # print(feature_maps.shape)
        # example_feature_map = feature_maps[0]
        # example_feature_map = example_feature_map.reshape(64, 64)
        # print("example_map", example_feature_map)
        # example_feature_map_np = example_feature_map.cpu().detach().numpy()
        # print("example_map_np", example_feature_map)


        # # Plot the feature map
        # # if len(example_feature_map_np) != 0:
        # plt.imshow(example_feature_map_2d, cmap='gray')  # Assuming grayscale feature maps
        # plt.show()
        
        batch_sentences = self.get_batch_sentences(tab)
        # # print("min", min(len(string) for string in batch_sentences))
        # # print("max", max(len(string) for string in batch_sentences))
        # # print("example", batch_sentences[0])
        
        # # change the dtype of the tabular data
        tab = tab.to(torch.float32)
        
        ind_to_keep = list(range(0, self.NUM_FEATURES))

        # ind_to_keep.remove(2)

        # ind_to_keep.remove(3)
        # ind_to_keep.remove(1)
        # ind_to_keep.remove(8)
        # ind_to_keep.remove(9)
        # ind_to_keep.remove(10)
        # ind_to_keep.remove(11)
        # ind_to_keep.remove(12)
        # ind_to_keep.remove(13)
        # ind_to_keep.remove(14)
        # ind_to_keep.remove(15)
        # ind_to_keep.remove(16)
        # ind_to_keep.remove(17)
        # ind_to_keep.remove(18)
        # ind_to_keep.remove(19)
        # ind_to_keep.remove(4)
        # ind_to_keep.remove(7)
        # ind_to_keep.remove(56)
        # ind_to_keep.remove(57)
        # ind_to_keep.remove(58)
        # ind_to_keep.remove(59)
        # ind_to_keep.remove(60)
        # ind_to_keep.remove(61)
        # ind_to_keep.remove(117)
        # ind_to_keep.remove(118)
        # ind_to_keep.remove(119)
        # ind_to_keep.remove(121)
        # ind_to_keep.remove(122)
        # ind_to_keep.remove(123)
        # ind_to_keep.remove(124)
        # ind_to_keep.remove(125)


        # Remove age and sex from tabular vector since we are using them as language model input
        tab_without_age_sex = tab[:,ind_to_keep]
        
        # # forward tabular data
        tab_without_age_sex = F.relu(self.fc1(tab_without_age_sex))
        

        language_inputs = self.tokenizer(batch_sentences, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        
        language_inputs = language_inputs.to('cuda')

        language_outputs = self.language_model(**language_inputs)

        # 1 x 768
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        pooled_states = language_outputs.pooler_output

        language_features_compressed = self.language_fc(pooled_states)

        # concat image, tabular data and data from language model
        #img, tab_without_age_sex, language_features_compressed
        x = torch.cat((img, tab_without_age_sex), dim=1)

        x = F.relu(self.fc2(x))

        out = self.fc3(x)

        out = torch.squeeze(out)
        # tab_without_age_sex_np = tab_without_age_sex.detach().cpu().numpy()
        # img_np = img.detach().cpu().numpy()
        # x_np = x.detach().cpu().numpy()

        # # Initialize the DeepExplainer
        # explainer = shap.DeepExplainer(model, data=[img_np, tab_without_age_sex_np, x_np])

        # # Compute Shapley values for a specific data point
        # data_point_idx = 0  # Index of the data point you want to explain
        # shap_values = explainer.shap_values([img_np[data_point_idx], tab_without_age_sex_np[data_point_idx], x_np[data_point_idx]])


        return out

    # def get_batch_sentences(self, tabular_to_encode):
    #     # return_tensors pt means pytorch
    #     tabular_to_encode = self.scaler.inverse_transform(tabular_to_encode.detach().cpu().numpy())

    #     batch_age = tabular_to_encode[:, 2]
    #     batch_sex = tabular_to_encode[:, 1]

    #     batch_sex_l = list(batch_sex)
    #     batch_age_l = list(batch_age.round(2))

    #     batch_sex_l = list(map(lambda x: 'female' if x == 0 else 'male', batch_sex_l))

    #     batch_pairs = list(zip(batch_sex_l, batch_age_l))

    #     batch_sentences = ["This subject is " + pair[0] + " and " + str(pair[1]) + " years old" for pair in batch_pairs]

    #     return batch_sentences

    def backtranslate_sentences(self, sentences, source_lang, target_lang):
        translate_client = translate.Client()

        translated_sentences = []

        for sentence in sentences:
            # Translate the sentence to the target language
            translation = translate_client.translate(
                sentence,
                target_language=target_lang
            )
            translated_text = translation['translatedText']

            # Translate the text back to the source language
            back_translation = translate_client.translate(
                translated_text,
                target_language=source_lang
            )
            backtranslated_text = back_translation['translatedText']

            translated_sentences.append(backtranslated_text)

        return translated_sentences

    def shuffle_sentences(self, paragraphs):
        shuffled_paragraphs = []
        for paragraph in paragraphs:
            sentences = paragraph.split('. ')
            # Split the paragraph into sentences
            random.shuffle(sentences)  # Shuffle the order of sentences
            shuffled_paragraph = '. '.join(sentences)  # Rejoin the shuffled sentences
            for _ in range (100):
                random.shuffle(sentences)
                temp = '. '.join(sentences)
                shuffled_paragraph += temp
            shuffled_paragraphs.append(shuffled_paragraph)
        return shuffled_paragraphs
    

    # Function to get synonyms of a word
    def get_synonyms(self, word):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name())
        return list(synonyms)

    # Function to perform synonym replacement in a sentence
    def synonym_replacement(self, sentence, num_replacements=5):
        words = nltk.word_tokenize(sentence)
        augmented_sentences = [sentence]

        for _ in range(num_replacements):
            selected_word = None
            while not selected_word:
                word_index = random.randint(0, len(words) - 1)
                selected_word = words[word_index]
                synonyms = self.get_synonyms(selected_word)

            if synonyms:
                synonym = random.choice(synonyms)
                words[word_index] = synonym

            augmented_sentences.append(" ".join(words))

        return augmented_sentences
    def augment_paragraphs(self, paragraphs, num_replacements=1):
        augmented_paragraphs = []
        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            augmented_sentences = []
            for sentence in sentences:
                augmented_sentences.extend(self.synonym_replacement(sentence, num_replacements))
            augmented_paragraphs.append(" ".join(augmented_sentences))
        return augmented_paragraphs

    def get_batch_sentences(self, tabular_to_encode):
        # return_tensors pt means pytorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tabular_to_encode = self.scaler.inverse_transform(tabular_to_encode.detach().cpu().numpy())
        # print(tabular_to_encode)
        # else: 
        #     tabular_to_encode = self.scaler.inverse_transform(tabular_to_encode.detach().cuda().numpy())
        batch_age = tabular_to_encode[:, 2]
        # print(batch_age)
        batch_sex = tabular_to_encode[:, 1]
        batch_cahalan_score = tabular_to_encode[:, 8]
        batch_excess_bl_drinking_2 = tabular_to_encode[:, 9]
        batch_lssaga_dsm4_youth_d04_diag = tabular_to_encode[:, 10]
        batch_lssaga_dsm4_youth_d05_diag = tabular_to_encode[:, 11]
        batch_highrisk_yss_extern = tabular_to_encode[:, 12]
        batch_highrisk_yss_intern = tabular_to_encode[:, 13]
        batch_highrisk_pss_extern = tabular_to_encode[:, 14]
        batch_highrisk_pss_intern = tabular_to_encode[:, 15]
        batch_highrisk_youthreport1_yfhi4 = tabular_to_encode[:, 16]
        batch_highrisk_youthreport1_yfhi3 = tabular_to_encode[:, 17]
        batch_highrisk_youthreport1_yfhi5 = tabular_to_encode[:, 18]
        batch_highrisk_youthreport1_leq_c_cnc = tabular_to_encode[:, 19]
        batch_hispanic = tabular_to_encode[:, 3]
        batch_race = tabular_to_encode[:, 4]
        batch_bmi = tabular_to_encode[:, 7]
        batch_rsq_problem_solving = tabular_to_encode[:, 56]
        batch_rsq_emotion_expression = tabular_to_encode[:, 57]
        batch_rsq_acceptance = tabular_to_encode[:, 58]
        batch_rsq_positive_thinking = tabular_to_encode[:, 59]
        batch_rsq_emotion_regulation = tabular_to_encode[:, 60]
        batch_rsq_cognitive_restructuring = tabular_to_encode[:, 61]

        # batch_age = tabular_to_encode["visit_age"]
        # batch_sex = tabular_to_encode["sex"]
        # batch_cahalan_score = tabular_to_encode["cahalan_score"]
        # batch_excess_bl_drinking_2 = tabular_to_encode["exceeds_bl_drinking_2"]
        # batch_lssaga_dsm4_youth_d04_diag = tabular_to_encode["lssaga_dsm4_youth_d04_diag"]
        # batch_lssaga_dsm4_youth_d05_diag = tabular_to_encode["lssaga_dsm4_youth_d05_diag"]
        # batch_highrisk_youthreport1_yfhi4 = tabular_to_encode["youthreport1_yfhi4"]
        # batch_highrisk_youthreport1_yfhi3 = tabular_to_encode["youthreport1_yfhi3"]
        # batch_highrisk_youthreport1_yfhi5 = tabular_to_encode["youthreport1_yfhi5"]
        # batch_highrisk_youthreport1_leq_c_cnc = tabular_to_encode["leq_c_cnc"]
        # batch_hispanic = tabular_to_encode["hispanic"]
        # batch_race = tabular_to_encode["race"]
        # batch_bmi = tabular_to_encode["bmi_zscore"]
        # batch_rsq_problem_solving = tabular_to_encode["rsq_problem_solving"]
        # batch_rsq_emotion_expression = tabular_to_encode["rsq_emotion_expression"]
        # batch_rsq_acceptance = tabular_to_encode["rsq_acceptance"]
        # batch_rsq_positive_thinking = tabular_to_encode["rsq_positive_thinking"]
        # batch_rsq_emotion_regulation = tabular_to_encode["rsq_emotion_regulation"]
        # batch_rsq_cognitive_restructuring = tabular_to_encode["rsq_cognitive_restructuring"]

        batch_youthreport2_shq1 = tabular_to_encode[:, 117]#"youthreport2_shq1"]
        batch_youthreport2_shq2 = tabular_to_encode[:, 118]#"youthreport2_shq2"]
        batch_youthreport2_shq3 = tabular_to_encode[:, 119]#"youthreport2_shq3"]
        batch_youthreport2_shq4 = tabular_to_encode[:, 120]#"youthreport2_shq4"]
        batch_youthreport2_shq5 = tabular_to_encode[:, 121]#"youthreport2_shq5"]

        batch_shq_weekday_sleep = tabular_to_encode[:, 122]#"shq_weekday_sleep"]
        batch_shq_weekend_sleep = tabular_to_encode[:, 123]#"shq_weekend_sleep"]
        batch_shq_weekend_bedtime_delay = tabular_to_encode[:, 124]#"shq_weekend_bedtime_delay"]
        batch_shq_weekend_wakeup_delay = tabular_to_encode[:, 125]#"shq_weekend_wakeup_delay"]
        # print(batch_youthreport2_shq1)
        batch_youthreport2_shq1 = list(batch_youthreport2_shq1)
        import datetime
        formatted_times = []
        # print(batch_youthreport2_shq1)
        for t in batch_youthreport2_shq1:
            if t <= 60:
                formatted_times.append("midnight")
            else:
                time_str = str(t)
                dot_index = time_str.index(".")
                time_str = time_str[:dot_index]
                if len(time_str) == 2:
                    time_str = "0" + time_str
                if len(time_str) == 3:
                    time_str = "0" + time_str
                # Extract the hour and minute components
                hour = int(time_str[:2])
                minute = int(time_str[2:])
                
                # Determine whether it's AM or PM based on the hour
                meridiem = "am" if hour < 12 else "pm"
                if hour >= 7:
                    meridiem = "pm"
                
                # Convert the hour to 12-hour format
                if hour >= 13:
                    hour -= 12
                
                # Format the time as a string
                formatted_time = "{:02d}:{:02d}{}".format(hour, minute, meridiem)
                formatted_times.append(formatted_time)
        batch_youthreport2_shq1 = formatted_times

        batch_youthreport2_shq2 = list(batch_youthreport2_shq2)
        formatted_times = []
        for t in batch_youthreport2_shq2:
            if t <= 60:
                formatted_times.append("midnight")
            else:
                time_str = str(t)
                dot_index = time_str.index(".")
                time_str = time_str[:dot_index]
                if len(time_str) == 3:
                    time_str = "0" + time_str
                # Extract the hour and minute components
                hour = int(time_str[:2])
                minute = int(time_str[2:])
                
                # Determine whether it's AM or PM based on the hour
                meridiem = "am" if hour < 12 else "pm"
                
                # Convert the hour to 12-hour format
                if hour >= 13:
                    hour -= 12
                
                # Format the time as a string
                formatted_time = "{:02d}:{:02d}{}".format(hour, minute, meridiem)
                
                formatted_times.append(formatted_time)
        batch_youthreport2_shq2 = formatted_times

        batch_youthreport2_shq3 = list(batch_youthreport2_shq3)
        formatted_times = []
        for t in batch_youthreport2_shq3:
            if t <= 60:
                formatted_times.append("midnight")
            else:
                time_str = str(t)
                dot_index = time_str.index(".")
                time_str = time_str[:dot_index]
                if len(time_str) == 2:
                    time_str = "0" + time_str
                if len(time_str) == 3:
                    time_str = "0" + time_str
                # Extract the hour and minute components
                hour = int(time_str[:2])
                minute = int(time_str[2:])
                
                # Determine whether it's AM or PM based on the hour
                meridiem = "am" if hour < 12 else "pm"
                
                # Convert the hour to 12-hour format
                if hour >= 13:
                    hour -= 12
                
                # Format the time as a string
                formatted_time = "{:02d}:{:02d}{}".format(hour, minute, meridiem)
                
                formatted_times.append(formatted_time)
        batch_youthreport2_shq3 = formatted_times

        batch_youthreport2_shq4 = list(batch_youthreport2_shq4)
        formatted_times = []
        for t in batch_youthreport2_shq4:
            if t <= 60:
                formatted_times.append("midnight")
            else:
                time_str = str(t)
                dot_index = time_str.index(".")
                time_str = time_str[:dot_index]
                if len(time_str) == 2:
                    time_str = "0" + time_str
                if len(time_str) == 3:
                    time_str = "0" + time_str
                # Extract the hour and minute components
                hour = int(time_str[:2])
                minute = int(time_str[2:])
                
                # Determine whether it's AM or PM based on the hour
                meridiem = "am" if hour < 12 else "pm"
                
                # Convert the hour to 12-hour format
                if hour >= 13:
                    hour -= 12
                
                # Format the time as a string
                formatted_time = "{:02d}:{:02d}{}".format(hour, minute, meridiem)
                
                formatted_times.append(formatted_time)
            
        batch_youthreport2_shq4 = formatted_times

        batch_youthreport2_shq5 = list(batch_youthreport2_shq5) # average sleep quality

        batch_shq_weekday_sleep = list(batch_shq_weekday_sleep)
        batch_shq_weekend_sleep = list(batch_shq_weekend_sleep)
        batch_shq_weekend_bedtime_delay = list(batch_shq_weekend_bedtime_delay)
        batch_shq_weekend_wakeup_delay = list(batch_shq_weekend_wakeup_delay)

        batch_shq_weekday_sleep = np.round(batch_shq_weekday_sleep, 2)
        batch_shq_weekend_sleep = np.round(batch_shq_weekend_sleep, 2)
        batch_shq_weekend_bedtime_delay = np.round(batch_shq_weekend_bedtime_delay, 2)
        batch_shq_weekend_wakeup_delay = np.round(batch_shq_weekend_wakeup_delay, 2)

        batch_support_comm_3 = tabular_to_encode[:, 51]#"youthreport2_chks_set2_chks3"]
        batch_support_comm_3_l = list(batch_support_comm_3)
        batch_support_comm_3_sentence = "there are supportive and caring adults at school."

        batch_support_comm_4 = tabular_to_encode[:, 52]#"youthreport2_chks_set2_chks4"]
        batch_support_comm_4_l = list(batch_support_comm_4)
        batch_support_comm_4_sentence = "there are supportive adults."

        batch_support_comm_5 = tabular_to_encode[:, 53]#"youthreport2_chks_set4_chks5"]
        batch_support_comm_5_l = list(batch_support_comm_5)
        batch_support_comm_5_sentence = "there are caring adults in the neighborhood."

        batch_support_comm_6 = tabular_to_encode[:, 54]#"youthreport2_chks_set4_chks6"]
        batch_support_comm_6_l = list(batch_support_comm_6)
        batch_support_comm_6_sentence = "there are trustworthy adults."

        batch_support_comm_7 = tabular_to_encode[:, 55]#"youthreport2_chks_set5_chks7"]
        batch_support_comm_7_l = list(batch_support_comm_7)
        batch_support_comm_7_sentence = "a part of clubs, sports teams, church/temple, or other group activities."

        batch_support_comm_8 = tabular_to_encode[:, 56]#"youthreport2_chks_set5_chks8"]
        batch_support_comm_8_sentence = "involved in music, art, literature, sports, or a hobby."
        batch_support_comm_8_l = list(batch_support_comm_8)

        batch_support_comm_9 = tabular_to_encode[:, 58]#"youthreport2_chks_set5_chks9"]
        batch_support_comm_9_sentence = "helpful towards other people."
        batch_support_comm_9_l = list(batch_support_comm_9)

        batch_support_comm_3_l = list(['generally does not feel like ' + batch_support_comm_3_sentence if 1 < x and x <= 2 
                                                else 'sometimes feels ' + batch_support_comm_3_sentence if  2 < x and x <= 3
                                                else 'generally does feel like ' + batch_support_comm_3_sentence if x >3 else "" for x in batch_support_comm_3_l])
        batch_support_comm_4_l = list(['generally does not feel like ' + batch_support_comm_4_sentence if 1 < x and x <= 2 
                                                else 'sometimes feels ' + batch_support_comm_4_sentence if  2 < x and x <= 3
                                                else 'generally does feel like ' + batch_support_comm_4_sentence if x >3 else "" for x in batch_support_comm_4_l])
        batch_support_comm_5_l = list(['generally does not feel like ' + batch_support_comm_5_sentence if 1 < x and x <= 2 
                                                else 'sometimes feels ' + batch_support_comm_5_sentence if  2 < x and x <= 3
                                                else 'generally does feel like ' + batch_support_comm_5_sentence if x >3 else "" for x in batch_support_comm_5_l])
        batch_support_comm_6_l = list(['generally does not feel like ' + batch_support_comm_6_sentence if 1 < x and x <= 2 
                                                else 'sometimes feels ' + batch_support_comm_6_sentence if  2 < x and x <= 3
                                                else 'generally does feel like ' + batch_support_comm_6_sentence if x >3 else "" for x in batch_support_comm_6_l])
        batch_support_comm_7_l = list(['generally does not feel like ' + batch_support_comm_7_sentence if 1 < x and x <= 2 
                                                else 'sometimes feels ' + batch_support_comm_7_sentence if  2 < x and x <= 3
                                                else 'generally does feel like ' + batch_support_comm_7_sentence if x >3 else "" for x in batch_support_comm_7_l])
        batch_support_comm_8_l = list(['generally does not feel ' + batch_support_comm_8_sentence if 1 < x and x <= 2 
                                                else 'sometimes feels ' + batch_support_comm_8_sentence if  2 < x and x <= 3
                                                else 'generally does feel ' + batch_support_comm_8_sentence if x >3 else "" for x in batch_support_comm_8_l])
        batch_support_comm_9_l = list(['generally does not feel ' + batch_support_comm_9_sentence if 1 < x and x <= 2 
                                                else 'sometimes feels ' + batch_support_comm_9_sentence if  2 < x and x <= 3
                                                else 'generally does feel ' + batch_support_comm_9_sentence if x >3 else "" for x in batch_support_comm_9_l])
        # end

        batch_sex_l = list(batch_sex)
        batch_age_l = list(np.round(batch_age, 2))
        batch_cahalan_score_l = list(batch_cahalan_score)
        batch_excess_bl_drinking_2_l = list(batch_excess_bl_drinking_2)
        batch_lssaga_dsm4_youth_d04_diag_l = list(batch_lssaga_dsm4_youth_d04_diag)
        batch_lssaga_dsm4_youth_d05_diag_l = list(batch_lssaga_dsm4_youth_d05_diag)
        # batch_highrisk_yss_extern_l = list(batch_highrisk_yss_extern)
        # batch_highrisk_yss_intern_l = list(batch_highrisk_yss_intern)
        # batch_highrisk_pss_extern_l = list(batch_highrisk_pss_extern)
        # batch_highrisk_pss_intern_l = list(batch_highrisk_pss_intern)
        batch_highrisk_youthreport1_yfhi4_l= list(batch_highrisk_youthreport1_yfhi4)
        batch_highrisk_youthreport1_yfhi3_l = list(batch_highrisk_youthreport1_yfhi3)
        batch_highrisk_youthreport1_yfhi5_l = list(batch_highrisk_youthreport1_yfhi5)
        batch_highrisk_youthreport1_leq_c_cnc_l = list(batch_highrisk_youthreport1_leq_c_cnc)


        batch_sex_l = list(map(lambda x: 'girl' if x == 0 else 'boy', batch_sex_l))
        batch_cahalan_score_l = list(map(lambda x: 'nondrinker' if x == 0 else "moderate drinker" if x ==1 else "heavy drinker" if x ==2 else "heavy binge drinker", batch_cahalan_score_l))
        batch_excess_bl_drinking_2_l = list(map(lambda x: "not a binge drinker" if x ==0 else "a binge drinker" if x ==1 else "a heavy marijuana user but not binker drinker" if x ==2 else "recently became a binge drinker" if x ==3 else "a former binge drinker who has since stopped binge drinking", batch_excess_bl_drinking_2_l))
        batch_lssaga_dsm4_youth_d04_diag_l = list(map(lambda x: "has been diagnosed with alcohol dependence disorder" if x ==1 else "has not been diagnosed with alcohol dependence disorder", batch_lssaga_dsm4_youth_d04_diag_l))
        batch_lssaga_dsm4_youth_d05_diag_l = list(map(lambda x:"has been diagnosed with marijuana dependence disorder" if x ==1 else "has not been diagnosed with marijuana dependence disorder", batch_lssaga_dsm4_youth_d05_diag_l))
        batch_highrisk_youthreport1_yfhi4_l = list(map(lambda x: "does not have a blood relative who has had problems with drugs" if x == 0 else "has a blood relative who has had problems with drugs", batch_highrisk_youthreport1_yfhi4_l))
        batch_highrisk_youthreport1_yfhi3_l = list(map(lambda x: "does not have a blood relative who has had problems with alcohol" if x == 0 else "has a blood relative who has had problems with alcohol", batch_highrisk_youthreport1_yfhi3_l))
        batch_highrisk_youthreport1_yfhi5_l = list(map(lambda x: "does not have a blood relative who saw visions or heard voices or thought people were spying on them or plotting against them" if x == 0 else "has a blood relative who saw visions or heard voices or thought people were spying on them or plotting against them", batch_highrisk_youthreport1_yfhi5_l))

        batch_pronoun_l = list(map(lambda x: 'She' if x == "girl" else 'He', batch_sex_l))

        batch_hispanic_l = list(batch_hispanic)
        batch_hispanic_l = list(map(lambda x: 'not Hispanic' if x == 0 else 'Hispanic', batch_hispanic_l))

        batch_race_l = list(batch_race)
        batch_race_l = list(map(lambda x: 'Native American' if x == 1 else
                                ("Asian" if x == 2 else
                                ("Pacific Islander" if x == 3 else
                                ("Black" if x == 4 else
                                    ("White" if x == 5 else
                                    ("Other race or race not specified"))))), batch_race_l))

        batch_bmi_l = list(batch_bmi)
        batch_bmi_l = list(round(l, 2) for l in batch_bmi_l)

        batch_rsq_problem_solving_l = list(batch_rsq_problem_solving)
        rsq_problem_solving_list = "\"I try to think of different ways to change the problem to fix the situation,\" "+ "\"I ask other people for help or ideas about how to make the problem better,\" " +"and " + "\"I do something to try to fix the problem or take action to change things.\" "
        batch_rsq_problem_solving_l = list(['mostly disagrees with sentiments like ' + rsq_problem_solving_list if 1 < x and x <= 2 
                                                else 'sometimes agrees and sometimes disagrees with sentiments like ' + rsq_problem_solving_list if  2 < x and x <= 3
                                                else 'mostly agrees with sentiments like ' + rsq_problem_solving_list if x >3 else "" for x in batch_rsq_problem_solving_l])

        batch_rsq_emotion_expression_l = list(batch_rsq_emotion_expression)
        rsq_emotion_expression_list = "\"I let someone or something know how I feel,\" "+ "\"I get sympathy, understanding, or support from someone (like a parent, friend, brother/sister, or teacher),\" " + "and " + "\"I let my feelings out (like by writing in my journal/diary, drawing/painting, complaining to let off steam, being sarcastic/making fun, listening to music, exercising, yelling, crying).\" "
        batch_rsq_emotion_expression_l = list(['mostly disagrees with sentiments like ' + rsq_emotion_expression_list if 1 < x <= 2 
                                                else 'sometimes agrees and sometimes disagrees with sentiments like ' + rsq_emotion_expression_list if  2 < x and x <= 3
                                                else 'mostly agrees with sentiments like ' + rsq_emotion_expression_list if x >3 else "" for x in batch_rsq_emotion_expression_l])

        batch_batch_rsq_acceptance_l = list(batch_rsq_acceptance)
        rsq_acceptance_list = "\"I decide I'm okay with the way I am, even though I'm not perfect,\" " + "\"I realize that I have to live with things the way they are,\" " + "and " + "\"I just take things as they are; I go with the flow.\" "
        batch_batch_rsq_acceptance_l = list(['mostly disagrees with sentiments like ' + rsq_acceptance_list if 1 < x <= 2 
                                                else 'sometimes agrees and sometimes disagrees with sentiments like ' + rsq_acceptance_list if  2 < x and x <= 3
                                                else 'mostly agrees with sentiments like ' + rsq_acceptance_list if x >3 else "" for x in batch_batch_rsq_acceptance_l])

        batch_rsq_positive_thinking_l = list(batch_rsq_positive_thinking)
        rsq_positive_thinking_list = "\"I tell myself that I can get through this, or that I'll do better next time,\" " + "\"I tell myself that everything will be alright,\" " + "and " + "\"I think of ways to laugh about it so that it won't seem so bad.\" "
        batch_rsq_positive_thinking_l= list(['mostly disagrees with sentiments like ' + rsq_positive_thinking_list if 1 < x <= 2 
                                                else 'sometimes agrees and sometimes disagrees with sentiments like ' + rsq_positive_thinking_list if  2 < x and x <= 3
                                                else 'mostly agrees with sentiments like' + rsq_positive_thinking_list if x >3 else "" for x in batch_rsq_positive_thinking_l])

        batch_rsq_emotion_regulation_l = list(batch_rsq_emotion_regulation)
        rsq_emotion_regulation_list = "\"I get help from others when I'm trying to figure out how to deal with my feelings,\" " + "\"I do something to calm myself down when I'm having problems with others (like take deep breaths, listen to music, pray, take a break, walk, meditate),\" " +"and " + "\"I keep my feelings under control when I have to, then let them out when they won't make things worse.\" "
        batch_rsq_emotion_regulation_l= list(['mostly disagrees with sentiments like ' + rsq_emotion_regulation_list if 1 < x <= 2 
                                                else 'sometimes agrees and sometimes disagree with sentiments like ' + rsq_emotion_regulation_list if  2 < x and x <= 3
                                                else 'mostly agrees with sentiments like ' + rsq_emotion_regulation_list if x >3 else "" for x in batch_rsq_emotion_regulation_l])

        batch_rsq_cognitive_restructuring_l = list(batch_rsq_cognitive_restructuring)
        rsq_cognitive_restructuring_list = "\" I tell myself that things could be worse, I tell myself that it doesn\'t matter, that it isn\'t a big deal, and I think about the things I\'m learning from the situation, or something good that will come from it. \" "
        batch_rsq_cognitive_restructuring_l = list(['mostly disagrees with sentiments like ' + rsq_cognitive_restructuring_list if 1 < x and x <= 2 
                                                    else 'sometimes agrees and sometimes disagrees with sentiments like ' + rsq_cognitive_restructuring_list if 2 < x and x <= 3 
                                                    else 'mostly agrees with sentiments like ' + rsq_cognitive_restructuring_list if x >3 else "" for x in batch_rsq_cognitive_restructuring_l])


        # add description of patients
        batch_pairs = list(zip(batch_sex_l, batch_age_l, batch_cahalan_score_l, batch_excess_bl_drinking_2_l, batch_lssaga_dsm4_youth_d04_diag_l, 
                                batch_lssaga_dsm4_youth_d05_diag_l, batch_highrisk_youthreport1_yfhi4_l, batch_highrisk_youthreport1_yfhi3_l,
                                batch_highrisk_youthreport1_yfhi5_l, batch_hispanic_l, batch_race_l, batch_bmi_l, batch_pronoun_l,
                                batch_rsq_problem_solving_l, batch_rsq_emotion_expression_l, batch_batch_rsq_acceptance_l, batch_rsq_positive_thinking_l,
                                batch_rsq_emotion_regulation_l, batch_rsq_cognitive_restructuring_l, batch_youthreport2_shq1, batch_youthreport2_shq2,
                                batch_youthreport2_shq3, batch_youthreport2_shq4, batch_youthreport2_shq5, batch_shq_weekday_sleep, batch_shq_weekend_sleep,
                                batch_shq_weekend_bedtime_delay , batch_shq_weekend_wakeup_delay, batch_support_comm_3_l, batch_support_comm_4_l, batch_support_comm_5_l,
                            batch_support_comm_6_l, batch_support_comm_7_l, batch_support_comm_8_l, batch_support_comm_9_l
        ))
        batch_sentences = ["This subject is a " + str(pair[1]) + " year old " + str(pair[0]) + " who is a " + str(pair[2]) + " and " + str(pair[3]) + ". " +
                            str(pair[12]) + " is " + str(pair[9]) + ", and " + str(pair[10]) + " with a BMI of " + str(pair[11]) + ". " +
                            str(pair[12]) + " goes to sleep at " + str(pair[19])+ " and wakes up at " + str(pair[21])+ " on weekdays, getting an average of " + str(pair[24]) + " hours of sleep on weekdays. "  +
                            str(pair[12]) + " goes to sleep at " + str(pair[20]) + " and wakes up at " + str(pair[22]) + " on weekends, getting an average of " +  str(pair[25]) + " hours of sleep on weekends. " +
                            "The weekend bedtime delay is " + str(pair[26]) + " and the weekend wakeup delay is " + str(pair[27])+ ". " + "On average, the sleep quality is " + str(pair[23])+ ". " +
                            str(pair[12]) + " " + str(pair[4]) + " and " + str(pair[5]) + " and " + str(pair[6]) + " and " + str(pair[7]) + " and " + str(pair[8]) + " "  +
                            str(pair[12]) + " " + str(pair[13]) + str(pair[12]) + " " + str(pair[14]) + str(pair[12]) + " " + str(pair[15]) + str(pair[12]) + " " + str(pair[16]) +
                            str(pair[12]) + " " + str(pair[17]) + str(pair[12]) + " " + str(pair[18]) + str(pair[12]) + " " + str(pair[28]) + " "+str(pair[12]) + " " + str(pair[29]) + " " + str(pair[12]) + " " +
                            str(pair[30]) +" " + str(pair[12]) + " " + str(pair[31]) + " " + str(pair[12]) + " " + str(pair[32]) + " " + str(pair[12]) + " " + str(pair[33]) +" "+ str(pair[12]) +
                            " " + str(pair[34]) +  " " for pair in batch_pairs]
        max_length = min(len(string) for string in batch_sentences)
        padded_strings = []
        for string in batch_sentences:
            truncated_string = string#[:400]
        #     padded_string = truncated_string.ljust(max_length, " ")
            padded_strings.append(truncated_string)


        # Example usage
        source_lang = "en"  # Source language is English
        target_lang = "fr"  # Target language is French

        # augmented_paragraphs = self.shuffle_sentences(batch_sentences)
        #choose between english and french randomly 
        # augmented_paragraphs = self.backtranslate_sentences(batch_sentences, source_lang, target_lang)
        
        augmented_paragraphs = self.augment_paragraphs(batch_sentences)
        #separately, then together
        return augmented_paragraphs

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 35, 50], gamma=0.1) # EMILY also try 0.1

        lr_scheduler = {
            'scheduler': scheduler,
            'name': 'lr_logging'
        }

        return [optimizer], [lr_scheduler]

    def training_step(self, batch, batch_idx):
        self.do_transform = True

        img, tab, y, subject_id = batch
        # img_cpu = img.detach().cpu().numpy()
        # print(subject_id)

        # print(y)
        # print("talking to the image")
        # img.clone().detach()
        # print(img.shape)
        # image_data = img.get_fdata()
        # transform = tio.RandomAffine(
        #     scales=(0.9, 1.2),
        #     degrees=10,
        # )
        # temp = img_cpu[0]
        # image_data_new = temp.reshape(1, 64, 64, 64)
        # transformed = transform(image_data_new)
        # img_cpu[0] = transformed[0]
        # img = torch.from_numpy(img_cpu)


        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

        loss = loss_func(y_pred, y.squeeze())

        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        self.train_results_df['subject'] = tuple(subject_id)
        self.train_results_df['label'] = y.squeeze().detach().cpu().numpy()
        self.train_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()

        tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
        self.train_results_df['age'] = tab_bef_normalization[:, 2]
        self.train_results_df['sex'] = tab_bef_normalization[:, 1]

        self.train_results_df_all = pd.concat([self.train_results_df_all, self.train_results_df], ignore_index=True)

        if BATCH_SIZE == 1:
            self.train_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.train_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            self.train_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)
            self.train_auc(torch.unsqueeze(y_pred_tag, 0), y)
            
            
    #     self.log('train_auc', self.val_auc, on_step=False, on_epoch=True)
        else:
            self.train_accuracy(y_pred_tag, y)

            self.train_macro_accuracy(y_pred_tag, y)
            self.train_macro_f1(y_pred_tag, y)
            self.train_auc(y_pred_tag, y)

        self.log('train_acc_step', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train_macro_acc_step', self.train_macro_accuracy, on_step=True, on_epoch=True)
        self.log('train_macro_f1', self.train_macro_f1, on_step=False, on_epoch=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True)
        # Log loss
        self.log('train_loss', loss, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        self.do_transform = False


        img, tab, y, subject_id = batch
        print(subject_id)
        print(y)
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

        loss = loss_func(y_pred, y.squeeze())
        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        self.val_results_df['subject'] = tuple(subject_id)
        self.val_results_df['label'] = y.squeeze().detach().cpu().numpy()
        self.val_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()

        tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
        self.val_results_df['age'] = tab_bef_normalization[:, 2]
        self.val_results_df['sex'] = tab_bef_normalization[:, 1]
        

        self.val_results_df_all = pd.concat([self.val_results_df_all, self.val_results_df], ignore_index=True)

        if BATCH_SIZE == 1:

            self.val_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.val_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
            self.val_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)
            self.val_auc(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.val_accuracy(y_pred_tag, y)

            self.val_macro_accuracy(y_pred_tag, y)
            self.val_macro_f1(y_pred_tag, y)
            self.val_auc(y_pred_tag, y)

        self.log('val_acc_step', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('val_macro_acc_step', self.val_macro_accuracy, on_step=True, on_epoch=True)
        self.log('val_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True)

        # Log loss
        self.log('val_loss', loss, on_step=True, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        self.do_transform = False

        img, tab, y, subject_id = batch
        y = y.to(torch.float32)

        y_pred = self(img, tab)

        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

        loss = loss_func(y_pred,
                         y.squeeze())  # loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze(), pos_weights = )

        y_pred_tag = torch.round(torch.sigmoid(y_pred))

        if BATCH_SIZE == 1:

            self.test_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

            self.test_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
        else:
            self.test_accuracy(y_pred_tag, y)

            self.test_macro_accuracy(y_pred_tag, y)

        self.log('test_acc_step', self.test_accuracy, on_step=True, on_epoch=False)
        self.log('test_macro_acc_step', self.test_macro_accuracy, on_step=True, on_epoch=True)

        self.log("test loss", loss)

        return loss

    def training_epoch_end(self, outs):

        filename_out = '/home/users/ewesel/results/train_out_language_' + str(
            self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.train_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)
        # log epoch metric
        self.log('train_acc_epoch', self.train_accuracy)
        self.log('train_macro_acc_epoch', self.train_macro_accuracy)
        self.log('train_macro_f1', self.train_macro_f1, on_step=False, on_epoch=True)
        self.log('train_auc', self.train_auc, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        # log epoch metric

        filename_out = '/home/users/ewesel/results/val_out_language_' + str(
            self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

        self.val_results_df_all.to_csv(filename_out)

        # Clear the dataframe so the new epoch can start fresh
        self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

        self.log('val_acc_epoch', self.val_accuracy)
        self.log('val_macro_acc_epoch', self.val_macro_accuracy)
        self.log('val_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True)
        self.log('val_auc', self.val_auc, on_step=False, on_epoch=True)


# import torch
# from torch import nn
# from pytorch_lightning.core.module import LightningModule
# from torch.nn import functional as F
# from monai.networks.nets import resnet10, resnet18, resnet34, resnet50
# from settings import IMAGE_SIZE, FEATURES, BATCH_SIZE, TARGET
# import torchmetrics
# import pandas as pd
# from transformers import AutoTokenizer, AutoModel
# import os

# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# class MultiModModelWithLanguage(LightningModule):
#     '''
#     Resnet Model Class including the training, validation and testing steps
#     '''

#     def __init__(self, class_weight, scaler):

#         super().__init__()

#         self.class_weight = class_weight

#         self.scaler = scaler

#         self.resnet = resnet10(pretrained=False,
#                                spatial_dims=3,
#                                num_classes=120,
#                                n_input_channels=1
#                                )

#         self.tokenizer = AutoTokenizer.from_pretrained('michiyasunaga/BioLinkBERT-base',
#                                                        cache_dir="/scratch/users/ewesel/")
#         self.language_model = AutoModel.from_pretrained('michiyasunaga/BioLinkBERT-base',
#                                                         cache_dir="/scratch/users/ewesel/")
#         med_alpaca_model_path = "/home/users/ewesel/dir/from_github/NCANDA_Contrastive/medAlpaca/medalpaca"
#         # self.tokenizer = AutoTokenizer.from_pretrained(med_alpaca_model_path)
#         # self.language_model = AutoModel.from_pretrained(med_alpaca_model_path)
                                                
#         # Freeze weights so those don't get trained        
#         # for param in self.language_model.parameters():
#         #     param.requires_grad = False


#         self.NUM_FEATURES = len(FEATURES)

#         # self.NUM_FEATURES = 0

#         # fc layer to make image size same as tabular data size
#         self.fc = nn.Linear(400, 1)

#         # combine resnet with final fc layer
#         self.imagenet = nn.Sequential(self.resnet, self.fc)
#         # fc layer that maps language model inputs to smaller dimension
#         self.language_fc = nn.Linear(768, 120)

#         # fc layer for tabular data. We substract 2 because age and sex are encoded as sentences
#         self.fc1 = nn.Linear((self.NUM_FEATURES - 2), 120)

#         # first fc layer which takes concatenated input
#         self.fc2 = nn.Linear((120 + 120 + 120), 32)

#         # final fc layer which takes concatenated imput
#         self.fc3 = nn.Linear(32, 1)

#         self.train_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

#         self.val_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

#         self.test_macro_accuracy = torchmetrics.Accuracy(task='multiclass', average='macro', num_classes=2)

#         self.train_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)

#         self.val_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)

#         self.test_accuracy = torchmetrics.Accuracy(task='multiclass', average='micro', num_classes=2)

#         self.train_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

#         self.train_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

#         self.val_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')

#         self.val_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

#         self.test_macro_f1 = torchmetrics.classification.MulticlassF1Score(task='multiclass', num_classes=2, average='macro')
        
#         self.test_auc = torchmetrics.classification.BinaryAUROC(task='multiclass', num_classes=2, average='macro')

#         self.results_column_names = ['subject', 'label', 'prediction', 'age', 'sex']

#         self.train_results_df = pd.DataFrame(columns=self.results_column_names)

#         self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)

#         self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)

#         self.val_results_df = pd.DataFrame(columns=self.results_column_names)

#     def forward(self, img, tab):
#         """

#         x is the input data

#         """
#         # run the model for the image
#         self.language_model = self.language_model.to('cuda')
#         # self.tokenizer = self.tokenizer
        
#         # print(img.shape)
#         img = torch.unsqueeze(img, 1)
#         img = img.to(torch.float32)
#         # print(img.type)
#         # print(img.shape)
#         img = self.resnet(img)
        
#         batch_sentences = self.get_batch_sentences(tab)
#         print(batch_sentences[0])
        
#         # change the dtype of the tabular data
#         tab = tab.to(torch.float32)
        
#         ind_to_keep = list(range(0, self.NUM_FEATURES))

#         ind_to_keep.remove(2)

#         ind_to_keep.remove(3)

#         # Remove age and sex from tabular vector since we are using them as language model input
#         tab_without_age_sex = tab[:,ind_to_keep]
        
#         # forward tabular data
#         tab_without_age_sex = F.relu(self.fc1(tab_without_age_sex))

#         language_inputs = self.tokenizer(batch_sentences, return_tensors="pt", padding = True, truncation = True)
        
#         language_inputs = language_inputs.to('cuda')
#         # print(language_inputs.keys())
#         print(language_inputs["input_ids"].shape, language_inputs["token_type_ids"].shape, language_inputs["attention_mask"].shape)
#         language_inputs["input_ids"] = language_inputs["input_ids"][:,:512]
#         language_inputs["token_type_ids"] = language_inputs["token_type_ids"][:,:512]
#         language_inputs["attention_mask"] = language_inputs["attention_mask"][:,:512]
        
#         language_outputs = self.language_model(**language_inputs)
#         # print(language_outputs)

#         # # 1 x 768
#         # # We "pool" the model by simply taking the hidden state corresponding
#         # # to the first token. We assume that this has been pre-trained
#         pooled_states = language_outputs.pooler_output

#         language_features_compressed = self.language_fc(pooled_states)

#         # concat image, tabular data and data from language model
#         x = torch.cat((img, tab_without_age_sex, language_features_compressed), dim=1)
#         # x = torch.cat((img, tab_without_age_sex), dim=1)

#         x = F.relu(self.fc2(x))

#         out = self.fc3(x)

#         out = torch.squeeze(out)

#         return out

    

    # def configure_optimizers(self):

    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    #     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 35, 50], gamma=0.8)

    #     lr_scheduler = {
    #         'scheduler': scheduler,
    #         'name': 'lr_logging'
    #     }

    #     return [optimizer], [lr_scheduler]

    # def training_step(self, batch, batch_idx):

    #     img, tab, y, subject_id = batch


    #     # img = torch.tensor(img).float()

    #     y = y.to(torch.float32)

    #     y_pred = self(img, tab)

    #     loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

    #     loss = loss_func(y_pred, y.squeeze())

    #     y_pred_tag = torch.round(torch.sigmoid(y_pred))

    #     # print("subj id", subject_id) # four 
    #     # print("df", self.train_results_df['subject']) # six, which is the batch size
    #     self.train_results_df['subject'] = tuple(subject_id)

        
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     #if device.type == "cpu":
    #     self.train_results_df['label'] = y.squeeze().detach().cpu().numpy()
    #     self.train_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()
    #     tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
    #     # else: 
    #     #     self.train_results_df['label'] = y.squeeze().detach().cuda().numpy()
    #     #     self.train_results_df['prediction'] = y_pred_tag.detach().cuda().numpy()
    #     #     tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cuda().numpy())
    #     self.train_results_df['age'] = tab_bef_normalization[:, 2]
    #     self.train_results_df['sex'] = tab_bef_normalization[:, 1]
    #     # print("about to train some cats")
    #     self.train_results_df_all = pd.concat([self.train_results_df_all, self.train_results_df], ignore_index=True)
    #     # print("train targets,", y)
    #     if BATCH_SIZE == 1:
    #         self.train_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

    #         self.train_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
    #         self.train_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)
    #         self.train_auc(torch.unsqueeze(y_pred_tag, 0), y)
              
    #     else:
    #         self.train_accuracy(y_pred_tag, y)

    #         self.train_macro_accuracy(y_pred_tag, y)
    #         self.train_macro_f1(y_pred_tag, y)
    #         self.train_auc(y_pred_tag, y)
        

    #     self.log('train_acc_step', self.train_accuracy, on_step=False, on_epoch=True)
    #     self.log('train_macro_acc_step', self.train_macro_accuracy, on_step=True, on_epoch=True)
    #     self.log('train_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True)
    #     self.log('train_auc', self.val_auc, on_step=False, on_epoch=True)
    #     # Log loss
    #     self.log('train_loss', loss, on_step=True, on_epoch=True)

    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     # print("val model language")

    #     img, tab, y, subject_id = batch
        
    #     y = y.to(torch.float32)

    #     y_pred = self(img, tab)

    #     loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

    #     loss = loss_func(y_pred, y.squeeze())
    #     y_pred_tag = torch.round(torch.sigmoid(y_pred))

    #     self.val_results_df['subject'] = tuple(subject_id)
    #     self.val_results_df['label'] = y.squeeze().detach().cpu().numpy()
    #     self.val_results_df['prediction'] = y_pred_tag.detach().cpu().numpy()

    #     tab_bef_normalization = self.scaler.inverse_transform(tab.detach().cpu().numpy())
    #     self.val_results_df['age'] = tab_bef_normalization[:, 2]
    #     self.val_results_df['sex'] = tab_bef_normalization[:, 1]
    #     # print("about to validate some cats")
    #     self.val_results_df_all = pd.concat([self.val_results_df_all, self.val_results_df], ignore_index=True)
    #     # print("targets val " , y)
    #     if BATCH_SIZE == 1:

    #         self.val_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

    #         self.val_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
    #         self.val_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)
    #         self.val_auc(torch.unsqueeze(y_pred_tag, 0), y)
    #     else:
    #         self.val_accuracy(y_pred_tag, y)
    #         self.val_macro_accuracy(y_pred_tag, y)
    #         self.val_macro_f1(y_pred_tag, y)
    #         self.val_auc(y_pred_tag, y)

    #     self.log('val_acc_step', self.val_accuracy, on_step=False, on_epoch=True)
    #     self.log('val_macro_acc_step', self.val_macro_accuracy, on_step=True, on_epoch=True)
    #     self.log('val_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True)
    #     self.log('val_auc', self.val_auc, on_step=False, on_epoch=True)

    #     # Log loss
    #     self.log('val_loss', loss, on_step=True, on_epoch=True)

    #     return loss

    # def test_step(self, batch, batch_idx):

    #     img, tab, y, subject_id = batch
    #     y = y.to(torch.float32)

    #     y_pred = self(img, tab)

    #     loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.class_weight).float())

    #     loss = loss_func(y_pred,
    #                      y.squeeze())  # loss = F.binary_cross_entropy(torch.sigmoid(y_pred), y.squeeze(), pos_weights = )

    #     y_pred_tag = torch.round(torch.sigmoid(y_pred))

    #     if BATCH_SIZE == 1:

    #         self.test_accuracy(torch.unsqueeze(y_pred_tag, 0), y)

    #         self.test_macro_accuracy(torch.unsqueeze(y_pred_tag, 0), y)
    #         self.test_macro_f1(torch.unsqueeze(y_pred_tag, 0), y)
    #         self.test_auc(torch.unsqueeze(y_pred_tag, 0), y)
    #     else:
    #         self.test_accuracy(y_pred_tag, y)

    #         self.test_macro_accuracy(y_pred_tag, y)
    #         self.test_macro_f1(y_pred_tag, y)
    #         self.test_auc(y_pred_tag, y)

    #     self.log('test_acc_step', self.test_accuracy, on_step=True, on_epoch=False)
    #     self.log('test_macro_acc_step', self.test_macro_accuracy, on_step=True, on_epoch=True)
    #     self.log('test_macro_f1', self.val_macro_f1, on_step=False, on_epoch=True)
    #     self.log('test_auc', self.val_auc, on_step=False, on_epoch=True)

    #     self.log("test loss", loss)

    #     return loss

    # def training_epoch_end(self, outs):
    #     filename_out = '/scratch/users/ewesel/train_out_language_'
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     if device.type == "cpu":
    #         filename_out = '/Users/emilywesel/Desktop/NCANDA/train_out_language_' 
    #     filename_out += str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

    #     self.train_results_df_all.to_csv(filename_out)

    #     # Clear the dataframe so the new epoch can start fresh
    #     self.train_results_df_all = pd.DataFrame(columns=self.results_column_names)
    #     # log epoch metric
    #     self.log('train_acc_epoch', self.train_accuracy)
    #     self.log('train_macro_acc_epoch', self.train_macro_accuracy)
    #     self.log('train_macro_f1', self.train_macro_f1)
    #     self.log('train_auc', self.train_auc)


    # def validation_epoch_end(self, outputs):
    #     # log epoch metric
    #     filename_out = '/scratch/users/ewesel/val_out_language_'
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     if device.type == "cpu":
    #         filename_out = '/Users/emilywesel/Desktop/NCANDA/val_out_language_' 
    #     filename_out += str(self.current_epoch) + '_' + TARGET + '_' + self.trainer.logger.experiment.name + '.csv'

    #     self.val_results_df_all.to_csv(filename_out)

    #     # Clear the dataframe so the new epoch can start fresh
    #     self.val_results_df_all = pd.DataFrame(columns=self.results_column_names)
    #     self.log('val_macro_f1', self.val_macro_f1)
    #     self.log('val_auc', self.val_auc)
    #     self.log('val_acc_epoch', self.val_accuracy)
    #     self.log('val_macro_acc_epoch', self.val_macro_accuracy)