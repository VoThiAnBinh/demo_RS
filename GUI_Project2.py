import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from underthesea import word_tokenize, pos_tag, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from surprise import Dataset, Reader,accuracy,BaselineOnly
from surprise.model_selection import train_test_split
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate

 # Đọc dữ liệu khách sạn
hotel_info = pd.read_csv('hotel_info.csv')
# Thay thế các giá trị NaN bằng "Unknown"
hotel_info = hotel_info.fillna("Unknown")
hotel_encoder=pd.read_csv("hotel_comment_encoded.csv")

# Hàm lấy các gợi ý dựa trên cosine similarity----------------------------
def get_recommendations_content_based(df, hotel_id, cosine_sim, nums=5):
    # Get the index of the hotel that matches the hotel_id
    matching_indices = df.index[df['Hotel_ID'] == hotel_id].tolist()
    if not matching_indices:
        st.write(f"No hotel found with ID: {hotel_id}")
        return pd.DataFrame()  # Return an empty DataFrame if no match
    idx = matching_indices[0]

    # Get the pairwise similarity scores of all hotels with that hotel
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort the hotels based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Get the scores of the nums most similar hotels (Ignoring the hotel itself)
    sim_scores = sim_scores[1:nums+1]
    # Get the hotel indices
    hotel_indices = [i[0] for i in sim_scores]
    # Return the top n most similar hotels as a DataFrame
    return df.iloc[hotel_indices]

# Hàm hiển thị các khách sạn gợi ý ra bảng chon hotel bất kỳ
def display_recommended_hotels(recommended_hotels, cols=5):
    for i in range(0, len(recommended_hotels), cols):
        cols = st.columns(cols)
        for j, col in enumerate(cols):
            if i + j < len(recommended_hotels):
                hotel = recommended_hotels.iloc[i + j]
                with col:
                    st.write(f"**Tên khách sạn:** {hotel['Hotel_Name']}")
                    st.write(f"**Địa chỉ:** {hotel['Hotel_Address']}")
                    st.write(f"**Điểm đánh giá:** {hotel['Total_Score']}")
                    st.write("-"*60)
                    st.write(f"**Vị trí:** {hotel['Location']}")
                    st.write(f"**Sạch sẽ:** {hotel['Cleanliness']}")
                    st.write(f"**Dịch vụ:** {hotel['Service']}")
                    st.write(f"**Tiện nghi:** {hotel['Facilities']}")
                    # st.write(f"**Đánh giá giá trị:** {hotel['Value_for_money']}")
                    # st.write(f"**Chất lượng phòng:** {hotel['Comfort_and_room_quality']}")

                    expander = st.expander("Thông tin khách sạn")
                    hotel_description = hotel['Hotel_Description']
                    truncated_description = ' '.join(hotel_description.split()[:100]) + '...'
                    expander.write(truncated_description)
                    expander.markdown("Nhấn vào mũi tên để đóng hộp text này.")

#Hàm làm sạch dữu liệu
STOP_WORD_FILE = 'vietnamese-stopwords.txt'
WRONG_WORD_FILE = 'wrong-word.txt'

# Đọc danh sách từ dừng
with open(STOP_WORD_FILE, 'r', encoding='utf-8') as file:
    stop_words = file.read().split('\n')

# Đọc danh sách từ sai
with open(WRONG_WORD_FILE, 'r', encoding='utf-8') as file:
    wrong_words = file.read().split('\n')

def preprocess_text(text, stop_words, wrong_words):
    # Chuyển thành chữ thường
    text = text.lower()

    # Loại bỏ ký tự số và ký tự đặc biệt
    text = re.sub(r'[0-9]', '', text)  # Loại bỏ số
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt (ngoài chữ cái và khoảng trắng)

    # Loại bỏ khoảng trắng dư thừa
    text = ' '.join(text.split())

    # Token hóa chuỗi
    tokens = word_tokenize(text)

    # Loại bỏ stop words và wrong words
    tokens = [t for t in tokens if t not in stop_words and t not in wrong_words]

    return ' '.join(tokens)  # Chuyển đổi danh sách token trở lại thành chuỗi

#tìm thông tin theo nội dung nhập(cosine)
def get_recommendations_cosine_from_searching(user_input, hotel_info_merge, vectorizer, tfidf_matrix, stop_words, wrong_words, num_recommendations=5):
    # Tiền xử lý văn bản người dùng
    user_text = preprocess_text(user_input, stop_words, wrong_words)

    # Chuyển đổi chuỗi văn bản của người dùng thành TF-IDF vector
    user_tfidf = vectorizer.transform([user_text])

    # Tính toán độ tương tự giữa chuỗi văn bản của người dùng và tất cả các khách sạn
    user_cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix).flatten()

    # Tạo ma trận tương đồng cosine cho từng khách sạn
    sim_scores = list(enumerate(user_cosine_sim))

    # Sắp xếp độ tương tự giảm dần
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Lấy top 5 khách sạn tương tự nhất
    sim_scores = sim_scores[:num_recommendations]

    # Lấy chỉ số của các khách sạn tương tự
    hotel_indices = [i[0] for i in sim_scores]

    recommendations = pd.DataFrame({
        'Hotel_ID': hotel_info_merge['Hotel_ID'].iloc[hotel_indices].values,
        'Hotel_Name': hotel_info_merge['Hotel_Name'].iloc[hotel_indices].values,
        'Hotel_Address': hotel_info_merge['Hotel_Address'].iloc[hotel_indices].values,
        'Hotel_Description': hotel_info_merge['Hotel_Description'].iloc[hotel_indices].values
    })

    return recommendations

#Load Model----------------------------------------------------------------
# Open and read file model
with open('cosine_model.pkl', 'rb') as f:
    cosine_sim_new = pickle.load(f)
# Tải vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Tải tfidf_matrix
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)
#Baseline-only
model_filename = 'baseline_model.pkl'
with open(model_filename, 'rb') as file:
    model_surprise = pickle.load(file)

#GUI-------------------------
# Thiết lập cấu hình trang
st.set_page_config(page_title="Hotel Recommendation System", layout="centered")

# Thiết lập tiêu đề và hiển thị đường kẻ
st.title("Đồ Án Tốt Nghiệp Data Science")
st.write("-"*60)


# Tạo thanh menu bên trái
menu = ["HOME", "Giới Thiệu Project", "Mô hình-Content-Based Filtering", "Mô hình-Collaborative Filtering","Phân Tích Xu Hướng"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'HOME':    
    # Thiết lập cấu hình trang
    st.write("### Học Viên: Võ Thị An Bình - Nguyễn Bảo Thi")
    st.write("-"*60)
    st.title("Hotel Recommendation System")
    st.subheader("Khám Phá Những Điểm Lưu Trú Tốt Nhất Dành Cho Bạn")

    # Hiển thị hình ảnh
    st.image("hotel-recommendation-systems.jpg", caption="Tiện Nghi Đẳng Cấp – Trải Nghiệm Dịch Vụ Vượt Trội – Một Kỳ Nghỉ Thư Giãn Đúng Nghĩa", use_column_width=True)
    st.write(
        """
       **Mục đích:**

       -Nhằm hỗ trợ người dùng tìm kiếm và lựa chọn những địa điểm lưu trú tối ưu, dựa trên sở thích và nhu cầu cá nhân.Giúp người dùng dễ dàng lựa chọn nơi nghỉ dưỡng phù hợp.

        **Lợi Ích Cho Người Dùng:**

        - **Gợi Ý Cá Nhân Hóa:** Nhận những đề xuất khách sạn phù hợp với sở thích và yêu cầu.
        - **Tiết Kiệm Thời Gian:** Tiết kiệm thời gian tìm kiếm
        - **Trải Nghiệm Tuyệt Vời:** Tận hưởng những dịch vụ và tiện nghi tốt nhất tại các điểm lưu trú, giúp kỳ nghỉ trở nên đáng nhớ hơn.

        """)
    st.write("-" * 60 + " Thank You " + "-" * 60)

elif choice == 'Giới Thiệu Project':  
    st.subheader("Giới Thiệu Project")
    st.write(
        """
        Mục tiêu của project nhằm xây dựng một ứng dụng thông minh giúp người dùng tìm kiếm và lựa chọn các sản phẩm hoặc dịch vụ phù hợp với sở thích và nhu cầu của khách hàng. 
        Hệ thống gợi ý sẽ sử dụng hai phương pháp phân tích dữ liệu chính để cung cấp các gợi ý cá nhân hóa, từ đó cải thiện trải nghiệm người dùng.

        """)

    with st.expander("Phương Pháp Xây Dựng Mô Hình"):
        st.write(
            """
            - **Content-Based Filtering**: Gợi ý dựa trên các đặc điểm của khách sạn như dịch vụ, tiện nghi, và loại phòng. Mô hình so sánh đặc điểm của khách sạn với sở thích người dùng để đưa ra các gợi ý tương tự.
            - **Collaborative Filtering**: Gợi ý dựa trên hành vi và sở thích của người dùng tương tự. Mô hình phân tích dữ liệu từ người dùng có sở thích giống nhau để đưa ra các gợi ý mới.
            """)
    
    st.image("https://www.researchgate.net/publication/337401660/figure/fig1/AS:827362874777603@1574270089942/The-principle-behind-collaborative-and-content-based-filtering-9-Pilah-Matur-App.ppm", use_column_width=True)



elif choice == 'Mô hình-Content-Based Filtering':
    st.subheader("Mô hình - Content-Based Filtering")
    st.write("-"*60)
    # Radio button cho phương pháp tìm kiếm
    search_method = st.radio("Chọn phương pháp tìm kiếm:",("Tìm Theo Khách Sạn", "Tìm Theo Nội Dung"))
    
    if search_method == "Tìm Theo Khách Sạn":
   
        # Lấy 10 khách sạn đầu tiên để gợi ý
        random_hotels = hotel_info.head(n=10)
        st.write("Danh sách 10 khách sạn gợi ý")
        st.write(random_hotels)

        st.session_state.random_hotels = random_hotels

        # Kiểm tra xem 'selected_hotel_id' đã có trong session_state hay chưa
        if 'selected_hotel_id' not in st.session_state:
            st.session_state.selected_hotel_id = None

        # Theo cách cho người dùng chọn khách sạn từ dropdown
        hotel_options = [(row['Hotel_Name'], row['Hotel_ID']) for index, row in st.session_state.random_hotels.iterrows()]

        # Tạo một dropdown với options là các tuple này
        selected_hotel = st.selectbox(
            "Chọn khách sạn",
            options=hotel_options,
            format_func=lambda x: x[0] ) # Hiển thị tên khách sạn
        

        # Cập nhật session_state dựa trên lựa chọn hiện tại
        st.session_state.selected_hotel_id = selected_hotel[1]

        if st.session_state.selected_hotel_id:
            st.write("Bạn đã chọn:", selected_hotel[0])
            st.write("Hotel_ID: ", st.session_state.selected_hotel_id)
            
            # Hiển thị thông tin khách sạn được chọn
            selected_hotel = hotel_info[hotel_info['Hotel_ID'] == st.session_state.selected_hotel_id]

            if not selected_hotel.empty:
                st.write('#### Bạn vừa chọn:')
                st.write('### ', selected_hotel['Hotel_Name'].values[0])

                hotel_description = selected_hotel['Hotel_Description'].values[0]
                truncated_description = ' '.join(hotel_description.split()[:100])
                st.write('##### Thông tin:')
                st.write(truncated_description, '...')

                st.write('##### Các khách sạn khác bạn cũng có thể quan tâm:')
                recommendations = get_recommendations_content_based(hotel_info, st.session_state.selected_hotel_id, cosine_sim=cosine_sim_new, nums=3)
                display_recommended_hotels(recommendations, cols=3)
            else:
                st.write(f"Không tìm thấy khách sạn với ID: {st.session_state.selected_hotel_id}")

    elif search_method == "Tìm Theo Nội Dung":
        user_input = st.text_area("Bạn muốn tìm khách sạn theo đặc điểm nào? (Vị trí, tiện nghi, giá cả,...):", "")
        st.write("Dưới đây là một số câu gợi ý:")
        st.markdown('<span style="opacity:0.5;">Ví dụ: khách sạn gần trung tâm thành phố,khách sạn ở khu vực yên tĩnh....</span>', unsafe_allow_html=True)
        if st.button("Nhận Gợi Ý"):
            if user_input.strip() == "":
                st.warning("Vui lòng nhập mô tả khách sạn.")
            else:
                recommendations = get_recommendations_cosine_from_searching(user_input, hotel_info, vectorizer, tfidf_matrix, stop_words, wrong_words)
                # Hiển thị kết quả gợi ý
                if not recommendations.empty:
                    st.subheader("Gợi Ý Khách Sạn:")
                    st.write(recommendations)
                else:
                    st.write("Không tìm thấy gợi ý nào phù hợp.")


##### Start
elif choice == 'Mô hình-Collaborative Filtering':
    st.subheader("Mô hình-Collaborative Filtering(SVD)")
    st.write("-"*60)

    # Display DataFrame
    data_show = pd.read_csv('data_show.csv')
    st.write("### Model Data")
    st.dataframe(data_show) 

    data_sub_1 = pd.read_csv('data_sub_1.csv')
    # Function to train and evaluate the model
    def train_and_evaluate_model(data):
        # Load data into Surprise
        reader = Reader(rating_scale=(1, 10))  # Define the rating scale if it's not explicitly defined
        data_surprise = Dataset.load_from_df(data[['Reviewer ID Encoded', 'Hotel ID Encoded', 'Score']], reader)
        # Use the SVD algorithm
        algo = SVD()
        # Evaluate performance using cross-validation
        cv_results = cross_validate(algo, data_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        # Train on the whole dataset
        trainset = data_surprise.build_full_trainset()
        algo.fit(trainset)    
        return algo, cv_results

    # Train and evaluate the model
    st.write("### Model Evaluation Results")
    algo, cv_results = train_and_evaluate_model(data_sub_1)
    
    # Display cross-validation results
    st.write("**Cross-Validation Results:**")
    st.write(f"**RMSE:** {cv_results['test_rmse'].mean():.4f}")
    st.write(f"**MAE:** {cv_results['test_mae'].mean():.4f}")

    # Function to get recommendations for a new ID
    def get_recommendations(new_id, data, algo):
        # Map New_ID to Reviewer ID Encoded
        new_id_to_encoded = data[['New_ID', 'Reviewer ID Encoded']].drop_duplicates()
        def get_reviewer_encoded(new_id):
            result = new_id_to_encoded[new_id_to_encoded['New_ID'] == new_id]
            if not result.empty:
                return result['Reviewer ID Encoded'].values[0]
            else:
                raise ValueError(f"New_ID '{new_id}' not found in data.")

        # Generate predictions
        reviewer_encoded = get_reviewer_encoded(new_id)
        hotel_ids = data[['Hotel ID', 'Hotel ID Encoded', 'Hotel_Name', 'Hotel_Address']].drop_duplicates()
        
        if 'Hotel ID Encoded' not in hotel_ids.columns:
            raise KeyError("'Hotel ID Encoded' column is missing from the DataFrame.")
        
        predictions = []
        for hotel_id_encoded in hotel_ids['Hotel ID Encoded']:
            pred = algo.predict(reviewer_encoded, hotel_id_encoded)
            predictions.append((hotel_id_encoded, pred.est))
        
        predictions_df = pd.DataFrame(predictions, columns=['Hotel_ID_Encoded', 'Estimated_Score'])
        recommendations = predictions_df.merge(hotel_ids, left_on='Hotel_ID_Encoded', right_on='Hotel ID Encoded')
        recommendations = recommendations[['Hotel ID', 'Hotel_Name', 'Hotel_Address', 'Estimated_Score']]
        top_recommendations = recommendations.sort_values(by='Estimated_Score', ascending=False).head(5)
        
        return top_recommendations

    # Input for new ID
    new_id = st.text_input('Enter Reviewer_ID', 'THI_8728.0_1.0')  # Replace default with a valid New_ID

    # Show recommendations if new ID is provided
    if new_id:
        try:
            st.write("### Top N Recommendations")
            top_recommendations = get_recommendations(new_id, data_sub_1, algo)
            st.dataframe(top_recommendations)  # Display the top recommendations
        except ValueError as e:
            st.error(str(e))  # Handle errors gracefully
        except KeyError as e:
            st.error(str(e))  # Handle errors gracefully

##### End

elif choice == 'Phân Tích Xu Hướng':
    st.write("Phân tích xu hướng")
    #  Đọc dữ liệu 
    data_comments = pd.read_csv('df_thongke.csv')
    # GroupName
    fig_m, ax = plt.subplots(figsize=(15, 6))
    sns.countplot(data=data_comments, x='Group Name', palette='viridis', ax=ax)
    ax.set_xlabel('Group Name')
    ax.set_ylabel('Count')
    ax.set_title('Số lượng của mỗi nhóm')
    st.pyplot(fig_m)

   # Vẽ biểu đồ phân bố số lượng chuyến du lịch theo tháng
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(data=data_comments, x='Month', palette='viridis', ax=ax)
    ax.set_xlabel('Tháng')
    ax.set_ylabel('Số lượng chuyến du lịch')
    ax.set_title('Xu hướng du lịch trong năm')
    ax.set_xticks(range(12))
    ax.set_xticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12'])
    st.pyplot(fig)

    #Xu hướng du lịch trong năm của từng nhóm 
    pivot_table = data_comments.groupby(['Group Name', 'Month']).size().reset_index(name='Count')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.barplot(data=pivot_table, x='Month', y='Count', hue='Group Name', palette='Set1', ax=ax)
    ax.set_title('Xu hướng du lịch trong năm của từng nhóm')
    ax.set_xlabel('Tháng')
    ax.set_ylabel('Số lượng bản ghi')
    ax.legend(title='Nhóm', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y')
    plt.tight_layout()
    st.pyplot(fig)

#    # Đếm số lần xuất hiện của mỗi quận
#     ward_counts = data_comments['Ward Name'].value_counts()
#     st.header('Phân bố số lượng của các quận')
#     fig, ax = plt.subplots(figsize=(12, 8))
#     ward_counts.plot(kind='bar', ax=ax, color='skyblue')
#     ax.set_title('Phân bố số lượng của các quận')
#     ax.set_xlabel('Tên quận')
#     ax.set_ylabel('Số lượng')
#     ax.set_xticklabels(ward_counts.index, rotation=90)
#     st.pyplot(fig)

