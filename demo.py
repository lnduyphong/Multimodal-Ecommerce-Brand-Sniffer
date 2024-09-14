import torch
import numpy as np
import pandas as pd
import re


from PIL import Image
from urllib import request as rq
from transformers import AutoModel, AutoTokenizer
from bs4 import BeautifulSoup

def parse_html(html):
    elem = BeautifulSoup(html, features="html.parser")
    text = ''
    for e in elem.descendants:
        if isinstance(e, str):
            text += e.strip()
        elif e.name in ['br',  'p', 'h1', 'h2', 'h3', 'h4','tr', 'th']:
            text += '\n'
        elif e.name == 'li':
            text += '\n- '
    return text


def preprocess_text(text):
    if not isinstance(text, str):
        return 'Nothing'
    if text.find('<div') != -1 or text.find('<span') != -1 or text.find('<br') != -1:
        text = parse_html(text)
    
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\sàáạãảâầấậẫẩăằắặẵẳèéẹẽẻêềếệễểìíịĩỉòóọõỏôồốộỗổơờớợỡởùúụũủưừứựữửỳýỵỹỷđ-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if len(text) > 800:
        text = text[:800]
    return text

def link2image(extra_imges):
    images = []
    for idx, img_url in enumerate(extra_imges):
        path = f'/home/lnduyphong/ecom/Resource/extra{idx+1}.jpg'
        rq.urlretrieve(img_url, path)
        image = Image.open(path).convert('RGB')
        images.append(image)
    return images

ecom_data = pd.read_excel('/home/lnduyphong/ecom/Data/all data.xlsx').sample(10)
ecom_data = ecom_data.drop(columns=['attributes', 'variants', 'categories', 'report_id', 'is_valid_report', 'report_name'])
ecom_data.dropna(axis='index', how='any', inplace=True, subset=['name', 'url_thumbnail'])
ecom_data.fillna('')
ecom_data.reset_index(inplace=True, drop=True)
ecom_data['name'] = ecom_data['name'].apply(preprocess_text)
ecom_data['description'] = ecom_data['description'].apply(preprocess_text)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True,
#     attn_implementation='sdpa', torch_dtype=torch.bfloat16)
# model = model.eval().to(device)
# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
model.eval()

rq.urlretrieve('https://cf.shopee.vn/file/b526ba1de367e5cdfe28fc774ad13f56', '/home/lnduyphong/ecom/Resource/image11.jpg')
rq.urlretrieve('https://cf.shopee.vn/file/309e7be5a6a6e2f047ff11d79e3633de', '/home/lnduyphong/ecom/Resource/image12.jpg')
rq.urlretrieve('https://cf.shopee.vn/file/b111c726eec2e34770f7e8cec1886927', '/home/lnduyphong/ecom/Resource/image21.jpg')
rq.urlretrieve('https://cf.shopee.vn/file/8a1032554fa920c59cf0c70cea6ea1d3', '/home/lnduyphong/ecom/Resource/image22.jpg')
rq.urlretrieve('https://vn-live-01.slatic.net/p/88ad65c29ba4d1504daff2b3f43b1d93.jpg', '/home/lnduyphong/ecom/Resource/image31.jpg')
rq.urlretrieve('https://vn-test-11.slatic.net/p/3c11dd259863500e9a067d0a30a892a4.jpg', '/home/lnduyphong/ecom/Resource/image32.jpg')
rq.urlretrieve('https://p16-oec-va.ibyteimg.com/tos-maliva-i-o3syd03w52-us/79ec45d64052426380357c1ddc45bd34~tplv-o3syd03w52-crop-webp:500:500.webp?from=2481794462', '/home/lnduyphong/ecom/Resource/image41.jpg')
rq.urlretrieve('https://p16-oec-va.ibyteimg.com/tos-maliva-i-o3syd03w52-us/e181ca5525a1442ab561071c1b16f56c~tplv-o3syd03w52-resize-jpeg:800:800.jpeg?from=1826719393', '/home/lnduyphong/ecom/Resource/image42.jpg')

brands = []

for i in range(len(ecom_data)):
    question = """
Task: Extract the Brand Name from Product Data

Instructions for the LLM:

- You are provided with the following product information: illustration images, a product title, and a detailed description.
- Your goal is to extract the brand name associated with the product, which refers to the name of the company or product line most closely tied to the product itself.
  - Do not extract distributor or seller names, as these are not part of the brand.
  
Approach:
1. Images: Carefully examine all images to identify any branding information. If there are multiple images, cross-reference them to find consistent branding details.
2. Text (Title & Description): Search the title and description for any mention of the brand. If the brand is present only in the text, extract it from there.
3. Consistency: Ensure the extracted brand name is consistent across all data sources where the brand is present. If different names appear, choose the name that appears most frequently across all sources and is most closely tied to the product, ignoring distributor names.
   
Special Instructions:
- If the brand appears in only one source (e.g., just in the title, description or just in one of the illustration images), extract it from that source.
- If no brand name is found in either the title, description, or images, return "No Brand".
"""

    path_img = '/home/lnduyphong/ecom/Resource/image_test.jpg'
    link_img = ecom_data.iloc[i]['url_thumbnail']
    extra_imges = str(ecom_data.iloc[i]['url_images']).replace(" ", '').replace("[", '').replace("]", '').replace("'", '').split(',')
    rq.urlretrieve(link_img, path_img)
    
    title1 = preprocess_text('FGU Bộ sản phẩm nước tẩy trang sạch sâu giàu khoáng dành cho da nhạy cảm La Roche Posay Micellar Water Ultra Sensitive S', True)
    des1 = preprocess_text('AO59SDBỘ SẢN PHẨM BAO GỒM01 x Nước làm sạch sâu và tẩy trang cho da nhạy cảm La Roche-Posay Micellar Water Ultra Sensitive Skin 400ml01 x Sữa Rửa Mặt Dành Cho Da Nhạy Cảm Effaclar - Gel Moussant Purifiant La Roche Posay 50ml*Hàng tặng kèm không bánTHÔNG TIN CHI TIẾT1. Nước làm sạch sâu và tẩy trang cho da nhạy cảm La Roche-Posay Micellar Water Ultra Sensitive Skin 400ml Với công nghệ cải tiến Glyco Micellar mang lại hiệu quả làm sạch sâu vượt trội, giúp lấy đi bụi bẩn, bã nhờn và lớp trang điểm nhưng vẫn an toàn cho làn da nhạy cảm & dễ kích ứng. Sản phẩm giàu nước khoáng La Roche-Posay với tính năng làm dịu da, giảm kích ứng và chống oxi hóa.Hướng dẫn hướng dẫn- Dùng bông cotton thoa sản phẩm lên mặt, mắt và môi. - Không cần rửa lại bằng nước.2. Sữa Rửa Mặt Dành Cho Da Nhạy Cảm Effaclar - Gel Moussant Purifiant La Roche Posay 50ml Có công thức được lựa chọn kĩ càng với các thành phần làm sạch dịu nhẹ phù hợp cho da dầu và da mụn nhạy cảm. Sản phẩm nhẹ nhàng loại bỏ độc tố cho da nhờ vào các hoạt chất làm sạch được chọn lọc cho làn da nhạy cảm, đồng thời loại bỏ bã nhờn dư thừa, mang lại làn da sạch và thoáng mát.Hướng dẫn sử dụng- Sử dụng hằng ngày vào buổi sáng và tối. Làm ẩm da với nước ấm, cho một lượng vừa đủ sản phẩm ra tay, tạo bọt, thoa sản phẩm lên mặt, tránh vùng da quanh mắt. Massage nhẹ nhàng, sau đó rửa sạch lại với nước và thấm khô.- Sau khi rửa mặt, nên cân bằng da và làm dịu da với nước xịt khoáng La Roche-Posay.THÔNG TIN THƯƠNG HIỆULa Roche-Posay là nhãn hàng dược mỹ phẩm đến từ Pháp trực thuộc tập đoàn L’Oreal đã hoạt động được hơn 30 năm, phối hợp nghiên cứu với các bác sĩ da liễu trên toàn thế giới cho ra đời các sản phẩm dưỡng da hướng đến thị trường sản phẩm dành cho da nhạy cảm, ngoài ra còn có dòng sản phẩm dành cho trẻ em. Thành phần nổi bật xuất hiện trong các sản phẩm của La Roche-Posay (LRP) là nước suối khoáng – thermal spring water. Tất cả những sản phẩm thuộc La Roche Posay đều được thử nghiệm lâm sàng và đánh giá khách quan từ bệnh viện Saint Jacques-Toulouse. Quy trình bào chế của sản phẩm cũng rất nghiêm ngặt mang lại cho người sử dụng vẻ đẹp tự nhiên và rất an toàn.Xuất xứ thương hiệu: PhápNơi sản xuất: PhápHạn sử dụng: 3 năm kể từ ngày sản xuất Ngày sản xuất: In trên bao bìThành phần: Xem chi tiết trên bao bì#LaRochePosay #Anthelios #chongnang #chinhhang24AO59')
    image11 = Image.open('/home/lnduyphong/ecom/Resource/image11.jpg').convert('RGB')
    image12 = Image.open('/home/lnduyphong/ecom/Resource/image12.jpg').convert('RGB')
    answer1 = 'La Roche Posay'
    
    title2 = preprocess_text('[N123] Mặt Nạ Dưỡng Da Chuyên Sâu cung cấp khoáng chất cần thiết cho da Soothing Mask Chăm Sóc Da Toàn Diện', True)
    des2 = preprocess_text('Mặt Nạ Dưỡng Da Chuyên Sâu cung cấp khoáng chất cần thiết cho da Soothing Mask Chăm Sóc Da Toàn Diện    Đối với nhiều chị em phụ nữ, mặt nạ giấy dường như trở thành “vật bất ly thân” phải có ở nhà hoặc thậm chí là đồ dùng cá nhân quen thuộc có trong túi xách bởi vì sự tiện lợi, dễ sử dụng. Mặt nạ giấy là bước bổ sung dưỡng chất hoàn hảo nhất cho làn da. Việc dùng mặt nạ 2,3 lần 1 tuần là cách tốt nhất để giữ cho da đủ ẩm và luôn tươi trẻ, rạng rỡ. Đây chính là bí quyết làm đẹp của nhiều bạn trẻ. Mặt nạ giấy vừa dưỡng da nhanh chóng, tiện lợi lại hiệu quả rõ rệt ngay lập tức nên bất cứ cô gái nào cũng nên có vài miếng mặt nạ trong tủ lạnh để thư giãn, dưỡng da nhé.1. MẶT NẠ Dưỡng Da Chuyên Sâu cung cấp khoáng chất cần thiết cho da - Cứu tinh cho da nhạy cảm, dễ kích ứng☺️🌞Mặt nạ Sleeping Mask kế thừa những hiệu quả “thanh lọc” làn da qua đêm từ dòng mặt nạ ngủ bán chạy nhất. 💁🏻 Sản phẩm giúp cấp ẩm và làm dịu mạnh vs thành phần rau má. 🔸Còn tăng khả năng sửa chữa tổn thương da. 🔸Tạo lớp màng dưỡng ẩm tối đa để ngăn mất nước. Giúp da trở nên mướt mịn và khỏe mạnh hơn. 🔸Thành phần nấm men rừng đặc biệt được hãng so sánh hiệu quả hơn 111,9% so với Madecasoides nhờ đặc tính: giàu chất chống oxy hoá. 🔸Giúp giảm thiểu những tổn thương do kích ứng, da mẩn đỏ và da nhạy cảm nhờ vào đặc tính chống viêm tự nhiên.🔸Công thức hoàn toàn có thể giúp tự cân bằng lượng dầu thừa và lượng nước trên da, giải quyết được kha khá vấn đề mà kem dưỡng chưa phát huy hết được.ĐẶC ĐIỂM NỔI BẬT:– Chất liệu mặt nạ 100% cotton giúp bạn khi đắp lên da sẽ thấy mịn màng đến không ngờ- Mặt nạ gắn vừa khít với khuôn mặt. Chỉ đến khi mặt nạ đã khô hẳn, dưỡng chất đã thẩm thấu hết vào da mặt bạn thì mặt nạ mới bị bong ra.– Khi đắp mặt nạ lên da, bạn sẽ cảm thấy mát lạnh và cảm nhận đc sự dễ chịu trên khuôn mặt. Khi lấy mặt nạ ra, bạn sẽ cảm nhận đc sự mịn màng và trắng sáng của làn da. HƯỚNG DẪN SỬ DỤNG VÀ BẢO QUẢN:- Rửa sạch mặt, thấm khô.- Lấy tấm mặt nạ ra trải rộng, dán ở khu vực cằm trước sau đó dán ở các vị trí khác như: mũi, trán, 2 bên má…- Để mặt nạ trên da khoảng 15 - 20 phút, mát xa nhẹ nhàng cho da được hấp thu dưỡng chất.- Bảo quản nơi khô ráo, thoáng mát, tránh ánh nắng trực tiếp và nhiệt độ cao.')
    image21 = Image.open('/home/lnduyphong/ecom/Resource/image21.jpg').convert('RGB')
    image22 = Image.open('/home/lnduyphong/ecom/Resource/image22.jpg').convert('RGB')
    answer2 = 'LIFTHENG'

    title3 = preprocess_text('HCMCHERRY PHARMACY Men Vi Sinh Fermentix hỗ trợ cân bằng hệ vi sinh hộp 12 lọ', True)
    des3 = ' '
    image31 = Image.open('/home/lnduyphong/ecom/Resource/image31.jpg').convert('RGB')
    image32 = Image.open('/home/lnduyphong/ecom/Resource/image32.jpg').convert('RGB')
    answer3 = 'Fermentix'

    title4 = preprocess_text('Set 10 khăn lau bàn Có thể giặt Nhà bếp Bền Hút nước Có thể giặt Có thể tái sử dụng Có thể tái sử dụng Có thể tái sử dụng Nhà bếp Bền Có thể giặt Hút nước Làm Sạch', True)
    des4 = ' '
    image41 = Image.open('/home/lnduyphong/ecom/Resource/image41.jpg').convert('RGB')
    image42 = Image.open('/home/lnduyphong/ecom/Resource/image42.jpg').convert('RGB')
    answer4 = 'No Brand'
    
    title_test = ecom_data.iloc[i]['name']
    des_test = ecom_data.iloc[i]['description']    
    image_test = Image.open(path_img).convert('RGB')
    
    if len(extra_imges) > 1:
        msgs = [
            {'role': 'user', 'content': [title1, des1, image11, image12, question]}, {'role': 'assistant', 'content': [answer1]},
            {'role': 'user', 'content': [title2, des2, image21, image22, question]}, {'role': 'assistant', 'content': [answer2]},
            {'role': 'user', 'content': [title3, des3, image31, image32, question]}, {'role': 'assistant', 'content': [answer3]},
            {'role': 'user', 'content': [title4, des4, image41, image42, question]}, {'role': 'assistant', 'content': [answer4]},
            {'role': 'user', 'content': [title_test, des_test, image_test, question]}
        ]
    else:
        msgs = [
            {'role': 'user', 'content': [title1, des1, image11, image12, question]}, {'role': 'assistant', 'content': [answer1]},
            {'role': 'user', 'content': [title2, des2, image21, image22, question]}, {'role': 'assistant', 'content': [answer2]},
            {'role': 'user', 'content': [title3, des3, image31, image32, question]}, {'role': 'assistant', 'content': [answer3]},
            {'role': 'user', 'content': [title4, des4, image41, image42, question]}, {'role': 'assistant', 'content': [answer4]},
            {'role': 'user', 'content': [title_test, des_test, image_test] + link2image(extra_imges[1:], '/home/lnduyphong/ecom/Resource/extra') + [question]}
        ]

    answer = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    brands.append(answer)

    print(f'Index: {i}')
    print(f'Title: {title_test}')
    print(f'Thumbnail Link: {link_img}')
    print(f'Description: {des_test}')
    if len(extra_imges) > 1:
        for k in range(1, len(extra_imges)):
            print(f'Image Link {k}: {extra_imges[k]}')
    print(f'LLM Answer: {answer}\n\n')

ecom_data['LLM_brand'] = brands
print(f'Exactly Match: {sum(brands == ecom_data.cleaned_brand) / len(ecom_data)}')
ecom_data.to_csv('/home/lnduyphong/ecom/Result/result.csv', index=False)