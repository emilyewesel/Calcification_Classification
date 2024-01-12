import torchvision.transforms as transforms
import torch

# IMAGE_PATH = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\images\init_test_images'
# IMAGE_PATH = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\images\nifti_images_tabular'
IMAGE_PATH = r'/Users/magdalinipaschali/Documents/NCANDA_T1_T2/T1_small'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_PATH = r'/home/groups/kpohl/ncanda-multi-modal/T1'

#print("making tabular")
if device.type =='cpu':
    IMAGE_PATH = r'/Users/emilywesel/Desktop/NCANDA/data/ncanda-multi-modal/T1'

# CSV_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\labels'
# CSV_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\labels\tabular_image_labels'
# CSV_FILE = r'/Users/magdalinipaschali/Documents/NCANDA_T1_T2/Tabular/full_per_visit_data_2021-03-26_processed cross_sectional_small.csv'

TABULAR_DATA_FILE = r'C:\Users\Selen\Desktop\LMU\multimodal_network\data\adni\tabular\adni_tabular_images'

CSV_FILE = r'/home/groups/kpohl/ncanda-multi-modal/Tabular/full_per_visit_data_2021-03-26_processed cross_sectional_scratch.csv'
if device.type =='cpu':
    CSV_FILE = r'/Users/emilywesel/Desktop/NCANDA/data/full_per_visit_data_2021-03-26_processed cross_sectional_scratch.csv'

FEATURES = ["cesd_score","sex","visit_age","hispanic","race","ses_parent_yoe","pds_score",
            "bmi_zscore","cahalan_score","exceeds_bl_drinking_2","lssaga_dsm4_youth_d04_diag","lssaga_dsm4_youth_d05_diag",
            "highrisk_yss_extern","highrisk_yss_intern","highrisk_pss_extern","highrisk_pss_intern","youthreport1_yfhi4",
            "youthreport1_yfhi3","youthreport1_yfhi5","leq_c_c","leq_c_cnc","leq_c_cnu","leq_c_dau","leq_c_dcu","leq_c_dnc",
            "leq_c_dnu","leq_c_dpc","leq_c_nc","leq_c_nu","leq_c_sn","leq_c_u","ctq_ea","ctq_en","ctq_minds","ctq_pa","ctq_pn",
            "ctq_sa","aces_total","tipi_agv","tipi_csv","tipi_ems","tipi_etv","tipi_ope","upps_nug","upps_pmt","upps_psv",
            "upps_pug","upps_sss","youthreport2_chks_set2_chks3","youthreport2_chks_set2_chks4","youthreport2_chks_set4_chks5",
            "youthreport2_chks_set4_chks6","youthreport2_chks_set5_chks7","youthreport2_chks_set5_chks8",
            "youthreport2_chks_set5_chks9","youthreport2_pwmkcr_involvement_pwmkcr3","rsq_problem_solving",
            "rsq_emotion_expression","rsq_acceptance","rsq_positive_thinking","rsq_emotion_regulation",
            "rsq_cognitive_restructuring","brief_inhibit_t","brief_beh_shift_t","brief_cog_shift_t","cnp_sfnb2_sfnb_tp",
            "cnp_sfnb2_sfnb_fp","cnp_sfnb2_sfnb_rtc","cnp_sfnb2_sfnb_tp0","cnp_sfnb2_sfnb_fp0","cnp_sfnb2_sfnb_rtc0",
            "cnp_sfnb2_sfnb_tp1","cnp_sfnb2_sfnb_fp1","cnp_sfnb2_sfnb_rtc1","cnp_sfnb2_sfnb_tp2","cnp_sfnb2_sfnb_fp2",
            "cnp_sfnb2_sfnb_tn0","cnp_sfnb2_sfnb_tn1","cnp_sfnb2_sfnb_tn2","cnp_sfnb2_sfnb_fn1","cnp_sfnb2_sfnb_fn2",
            "cnp_sfnb2_sfnb_rtc2","cnp_sfnb2_sfnb_mcr","cnp_sfnb2_sfnb_mrtc","cnp_sfnb2_sfnb_mrt","cnp_sfnb2_sfnb_meff",
            "cnp_er40d_er40ang","cnp_er40d_er40fear","cnp_er40d_er40hap","cnp_er40d_er40noe","cnp_er40d_er40sad",
            "cnp_er40d_er40_fpa","cnp_er40d_er40_fpf","cnp_er40d_er40_fph","cnp_er40d_er40_fpn","cnp_er40d_er40_fps",
            "cnp_er40d_er40angrt","cnp_er40d_er40fearrt","cnp_er40d_er40haprt","cnp_er40d_er40noert","cnp_er40d_er40sadrt",
            "dd100_logk_1d","dd100_logk_7d","dd100_logk_1mo","dd100_logk_6mo","dd1000_logk_1d","dd1000_logk_7d","dd1000_logk_1mo",
            "dd1000_logk_6mo","np_ehi_result","stroop_total_mean","stroop_stroopm_rr_diffrt","stroop_conm_rr_mean",
            "stroop_incm_rr_mean","youthreport2_shq1","youthreport2_shq2","youthreport2_shq3","youthreport2_shq4",
            "youthreport2_shq5","shq_weekday_sleep","shq_weekend_sleep","shq_weekend_bedtime_delay","shq_weekend_wakeup_delay",
            "shq_sleepiness","shq_circadian","stroop_error_sum","stroop_miss_sum"]

TARGET = 'depressive_symptoms'

IMAGE_SIZE = 64

NUM_FEATURES = len(FEATURES)

BATCH_SIZE = 8# must be a multiple of four

TRAIN_SIZE = 1
VAL_SIZE = 1
TEST_SIZE = 1

# transformation for the input images
transformation = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor()
])

# transformation for the labels
target_transformations = None
