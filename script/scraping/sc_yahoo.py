import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from queue import Queue
from time import sleep

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


# selenium網頁設置
def setup_driver():
    my_options = webdriver.ChromeOptions()
    my_options.add_argument("--start-maximized")
    my_options.add_argument("--incognito")
    my_options.add_argument("--disable-popup-blocking")
    my_options.add_argument("--disable-notifications")
    my_options.add_argument("--lang=zh-TW")
    return webdriver.Chrome(options=my_options)


# 網頁滾動
def scrollWeb(driver):
    innerHeight = 0
    offset = 0
    count = 0
    limit = 1

    while count <= limit:
        offset = driver.execute_script("return document.documentElement.scrollHeight;")

        driver.execute_script(
            f"""
                window.scrollTo({{
                    top: {offset},
                    behavior: 'smooth'
                }})
            """
        )

        sleep(2)

        innerHeight = driver.execute_script(
            "return document.documentElement.scrollHeight;"
        )

        if offset == innerHeight:
            count += 1


# 商品連結獲取
def getItemURL(driver, linkData):
    item_link = driver.find_elements(
        By.CSS_SELECTOR, "div.sc-1r6nvi-2.fKYvGI > a.sc-1r6nvi-4"
    )

    for i in item_link:
        link = i.get_attribute("href")
        linkData.append({"link": link})


# 確認是否有下頁按鈕
def findNextPage(driver):
    try:
        WebDriverWait(driver, 2.5).until(
            EC.presence_of_element_located(
                (
                    By.CSS_SELECTOR,
                    "a.Pagination__arrowBtn___2ihnp.Pagination__icoArrowRight___2KprV.Pagination__hideArrowOnMobile___2HsbF.Pagination__button___fFc6Y",
                )
            )
        )
        return True
    except TimeoutException:
        return False


# 前往下頁
def GoNextPage(driver):
    driver.find_element(
        By.CSS_SELECTOR,
        "a.Pagination__arrowBtn___2ihnp.Pagination__icoArrowRight___2KprV.Pagination__hideArrowOnMobile___2HsbF.Pagination__button___fFc6Y",
    ).click()


# 取得商品內文
def getInfo(driver, link):
    try:
        driver.get(link)
        sleep(1)

        item_Name = driver.find_element(
            By.CSS_SELECTOR, "h1.title__dV9Gh"
        ).get_attribute("innerText")

        try:
            item_Price = driver.find_element(
                By.CSS_SELECTOR, "em.price__cJlVD"
            ).get_attribute("innerText")
        except:
            item_Price = driver.find_element(
                By.CSS_SELECTOR, "em.price__s1coi"
            ).get_attribute("innerText")

        try:
            WebDriverWait(driver, 1).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "button.ExpanderPanel__MoreExpander-sc-cjbz2l-1")
                )
            )

            button = driver.find_element(
                By.CSS_SELECTOR, "button.ExpanderPanel__MoreExpander-sc-cjbz2l-1"
            )

            button.click()
            sleep(0.5)

            item_Info = driver.find_element(
                By.CSS_SELECTOR, "yec-cocoon.itemInfo__6y0S6"
            ).text

        except TimeoutException:
            item_Info = driver.find_element(
                By.CSS_SELECTOR, "yec-cocoon.itemInfo__6y0S6"
            ).text

        return {"itemName": item_Name, "itemPrice": item_Price, "itemInfo": item_Info}
    except Exception as e:
        print(f"Error processing link {link}: {e}")
        return None


# 多執行緒
def process_chunk(links):
    driver = setup_driver()
    results = []
    try:
        for item in links:
            link = item.get("link")
            result = getInfo(driver, link)
            if result:
                results.append(result)
    finally:
        driver.quit()
    return results


# 商品內文存檔
def save_json(folderPath, filename, data):
    with open(f"{folderPath}/{filename}", "w", encoding="utf-8") as f:
        json.dump(data, ensure_ascii=False, indent=4, fp=f)


# 主執行程式
def main():
    folderPath = "data/yahooMarket_" + datetime.now().strftime("%Y%m%d")
    if not os.path.exists(folderPath):
        os.makedirs(folderPath)

    # 第一步：獲取所有連結
    linkData = []
    driver = setup_driver()
    try:
        driver.get(
            "https://tw.bid.yahoo.com/tw/%E6%89%8B%E6%A9%9F%E7%A9%BA%E6%A9%9F-23960-category.html?%3Bacu=3%3Bdisp%3Dlist%3Bpg%3D%7Bpage%7D%3Brefine%3Dcon_used%3Bsort%3D-ptime&acu=0&disp=list&sort=-ptime"
        )
        print("開始爬取連結 . . .")

        while True:
            scrollWeb(driver)
            sleep(1)
            getItemURL(driver, linkData)
            if findNextPage(driver):
                GoNextPage(driver)
            else:
                break
    finally:
        driver.quit()

    # 過濾掉搜索頁面的連結
    linkData = [item for item in linkData if "search" not in item["link"]]
    save_json(folderPath, "yahooMarket_link.json", linkData)
    print(f"已取得 {len(linkData)} 筆連結")

    # 將連結分成多個chunk
    num_workers = 4
    chunk_size = len(linkData) // num_workers + 1
    chunks = [linkData[i : i + chunk_size] for i in range(0, len(linkData), chunk_size)]

    # 使用線程池同時處理多個chunk
    print("開始處理詳細資訊...")
    all_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk): i for i, chunk in enumerate(chunks)
        }

        for future in as_completed(future_to_chunk):
            chunk_num = future_to_chunk[future]
            try:
                results = future.result()
                all_results.extend(results)
                print(f"完成第 {chunk_num + 1} 組資料，目前總計 {len(all_results)} 筆")
            except Exception as e:
                print(f"處理第 {chunk_num + 1} 組資料時發生錯誤: {e}")

    print("詳細資訊取得完畢")
    save_json(folderPath, "yahooMarket_phone.json", all_results)
    print("資料寫入完畢")
    print(f"檔案位置: {folderPath}/yahooMarket_phone.json")
    print(f"已取得 {len(all_results)} / {len(linkData)} 筆連結")


if __name__ == "__main__":
    main()
