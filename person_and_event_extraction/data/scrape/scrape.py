import selenium
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import pandas as pd

driver_path ="D:\\chromedriver_win32\\chromedriver.exe"
options = selenium.webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
driver = selenium.webdriver.Chrome(driver_path, options=options)
web_url = "https://www.cnnindonesia.com/nasional/politik"

political_figures = [
    ["Megawati", "Megawati Soekarnoputri"],
    ["SBY", "Susilo Bambang Yudhoyono"],
    ["Risma", "Tri Rismaharini"],
    ["Anies", "Anies Baswedan"],
    ["Luhut", "Luhut Binsar Pandjaitan"],
    ["Ganjar", "Ganjar Pranowo"],
    ["Erick Thohir"],
    ["Puan", "Puan Maharani"],
    ["Surya Paloh"],
]

political_figures_url = {}

i_pol = 0
while i_pol < len(political_figures):
    figure = political_figures[i_pol]
    main_name = figure[0]

    driver.get(web_url)

    print("Searching for " + main_name + "...")

    search = driver.find_element_by_name("query")
    search.clear()
    search.send_keys(main_name)
    search.send_keys(Keys.RETURN)

    urls = []

    try:
        for _ in range(5):
            list_of_news = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "media_rows"))
            )
            
            articles = list_of_news.find_elements_by_tag_name("article")
            for article in articles:
                anchor = WebDriverWait(article, 100).until(
                    EC.presence_of_element_located((By.TAG_NAME, "a"))
                )
                # anchor = article.find_element_by_tag_name("a")
                urls.append(anchor.get_attribute("href"))
            
            next_button = driver.find_element_by_class_name("next")
            next_button.click()
            print("NEXT PAGE")
            i_pol += 1
        
        political_figures_url[main_name] = urls
    except Exception as e:
        print(e)

data = []    

print("SUDAHHH")

for k in political_figures_url.keys():
    i = 0
    while i < len(political_figures_url[k]):
        url = political_figures_url[k][i]
    # for url in political_figures_url[k]:
        new_data_entry = {
            "person": k,
        }
        driver.get(url)
        try:
            content_detail = WebDriverWait(driver, 100).until(
                EC.presence_of_element_located((By.CLASS_NAME, "content_detail"))
            )

            title = content_detail.find_element_by_tag_name("h1").text

            new_data_entry["title"] = title

            if title.startswith("VIDEO"):
                print("LAH MASUK VIDEO")
                i += 1
                continue

            # get id detikdetailtext
            detikdetailtext = content_detail.find_element_by_id("detikdetailtext")

            # get all p tag
            paragraphs = detikdetailtext.find_elements_by_tag_name("p")
            # get all the text
            text = ""
            for paragraph in paragraphs:
                text += paragraph.text
            
            new_data_entry["content"] = text

            data.append(new_data_entry)
            i += 1
        except Exception as e:
            print(e)

driver.quit()

dataset = pd.DataFrame(data)
dataset.to_csv("data.csv", index=False)