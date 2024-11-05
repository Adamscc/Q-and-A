from transformers import AutoTokenizer, AutoModelForCausalLM, T5ForConditionalGeneration, T5Tokenizer, \
    BertTokenizerFast, AutoModelForTokenClassification
import torch.nn.functional as F
from peft import PeftModel
import requests
import urllib.request
import urllib.parse
from lxml import etree
from bs4 import BeautifulSoup
import random
import warnings
from urllib3.exceptions import InsecureRequestWarning
import re
import torch
import json
from py2neo import Graph, Node, Relationship, NodeMatcher
import wikipedia
from tqdm import tqdm
import time
import opencc
from firecrawl import FirecrawlApp


class TextProcessor:
    @staticmethod
    def remove_repeated(input_str):
        # 实现移除重复字符的逻辑
        seen = set()
        result = []

        for char in input_str:
            if char not in seen:
                seen.add(char)
                result.append(char)

        return ''.join(result)

    @staticmethod
    def convert_to_simplified(chinese_text):
        # 实现繁体转简体的逻辑
        converter = opencc.OpenCC('t2s')  # 繁体到简体：t2s
        simplified = converter.convert(chinese_text)
        return simplified

    @staticmethod
    def format_entity(text):
        outs = re.sub(r'<.*?>', '', text)
        outs = re.sub(r',\s*]', ']', outs)

        outs = re.search(r'\[\s*{.*?}\s*\]', outs, re.S).group(0)
        outs = re.sub(r',\s*]', ']', outs)
        result = json.loads(outs)
        result = [item for item in result if item["label"] != "Date" and item["label"] != "Identity"]
        return result

    @staticmethod
    def Ran_sele(str, maxnum=500):
        str = "".join(str.split())
        max_start_index = max(len(str) - maxnum, 0)
        start_index = random.randint(0, max_start_index)
        return str[start_index:start_index + maxnum]

    @staticmethod
    def format_an(text):
        # 找到第一个标记的位置
        star_index = text.find('<end_of_turn>')

        if star_index != -1:
            # 提取从第一个标记到字符串末尾的内容
            result = text[star_index:]
        else:
            result = ""  # 如果没有找到标记，可以返回空字符串或其他处理方式
        result = result.replace("<end_of_turn>", "")
        result = re.sub("\n+", "\n", result)
        return result

    @staticmethod
    def split_text(text, max_length=300):
        # 初始化分段列表
        segments = []

        # 循环遍历文本，按最大长度切割
        while len(text) > max_length:
            # 找到第一个300字符以内的段落
            split_index = max_length

            split_index = max_length

            # 确保分割点不会超出文本长度
            split_index = min(split_index, len(text))

            # 将切分好的段落添加到分段列表
            segments.append(text[:split_index].strip())
            # 更新剩余的文本
            text = text[split_index:].strip()

        # 将最后一部分文本添加到分段列表
        if text:
            segments.append(text)

        return segments

    @staticmethod
    def extract_surrounding_text(text, keyword, char_range=250):
        # 找到关键字的位置
        keyword_position = text.find(keyword)

        if keyword_position == -1:
            return f"关键字 '{keyword}' 未在文本中找到。"

        # 计算前后500个字符的位置
        start_position = max(0, keyword_position - char_range)
        end_position = min(len(text), keyword_position + len(keyword) + char_range)

        # 提取文本
        surrounding_text = text[start_position:end_position]

        return surrounding_text

    @staticmethod
    def format_relations(relations):
        pattern = r"\[(.*?)\(Node\('PERSON', name='(.*?)'\), Node\('(\w+)', name='(.*?)'\)\)\]"
        matches = re.findall(pattern, relations)
        result = "\n".join(["{}和{}的关系是{}".format(match[1], match[3], match[0]) for match in matches])
        relations_list = ["{}到{}的关系是{}".format(match[3], match[1], match[0]) for match in matches]
        return relations_list

    @staticmethod
    def format_text_from_url(text):
        text = re.sub(r'\(https?://[^\s)]+\)', '', text)
        text = re.sub(r'\(http?://[^\s)]+\)', '', text)
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'<.*?>', ' ', text).strip()
        rep = {
            "\n": "",
            " ": "",
            "(javascript:void(0))": "",
            "-": "",
            "!": "",
            "*": "",
            "#": "",
            "/": "",
            "|": "",
            ":": "",
            "~": "",
            "\\": "",
            "]": "",
            ")": "",
            ">": ""
        }
        pattern = re.compile("|".join(re.escape(key) for key in rep.keys()))
        text = pattern.sub(lambda x: rep[x.group()], text)
        return text


class WikipediaHandler:
    @staticmethod
    def wiki(key_words):
        # 实现维基百科搜索的逻辑
        wikipedia.set_lang("zh")
        search_results = wikipedia.search(key_words)

        if search_results:
            try:
                # 尝试获取第一个搜索结果的页面内容
                page = wikipedia.page(search_results[0])
                title = TextProcessor.convert_to_simplified(page.title)
                summary = TextProcessor.convert_to_simplified(page.summary)
                content = TextProcessor.convert_to_simplified(page.content)
                content = re.sub("\s+", "\n", content)
                content = re.sub("\n+", "\n", content).replace("=", "")[:1000]
                return f"标题：{title}\n文章内容：{content}"

            except wikipedia.exceptions.DisambiguationError as e:
                # 如果遇到歧义页，列出所有可能的选项
                options = e.options
                # 自动选择第一个选项，或者提示用户选择
                print("出现歧义，请从以下词条中选择一个：")
                for option in options:
                    print(f"{options.index(option) + 1}. {option}")
                choice = eval(input("请输入序号："))
                selected_option = options[choice - 1]
                page = wikipedia.page(selected_option)
                title = TextProcessor.convert_to_simplified(page.title)
                summary = TextProcessor.convert_to_simplified(page.summary)
                content = TextProcessor.convert_to_simplified(page.content)[:1000]
                content = re.sub("\s+", "\n", content)
                content = re.sub("\n+", "\n", content).replace("=", "")
                return f"标题：{title}\n文章内容：{content}"
            except wikipedia.exceptions.PageError:
                return "未找到该信息"

        else:
            return "未找到该信息"


class Firecrawl:
    def __init__(self, api_key):
        self.app = FirecrawlApp(api_key=api_key)

    def get_text_from_url(self, url):
        scraped_data = self.app.scrape_url(url)
        text = TextProcessor.format_text_from_url(scraped_data['markdown'])
        return text

class KnowledgeGraph:
    def __init__(self, ner, agent, db_url, auth):
        self.db = Graph(db_url, auth=auth)
        self.matcher = NodeMatcher(self.db)
        self.agent = agent
        self.ner = ner

    def rel_query(self, q, limit=10, depth=2):
        # 实现关系查询的逻辑
        node = self.matcher.match(name=q[0]["entity"]).first()
        rel = ""
        if node:
            if limit != 0:
                query = f"""
                       MATCH (n)-[r*1..{depth}]-(m)
                       WHERE id(n) = $node_id
                       RETURN m, r
                       LIMIT {limit};
                       """
            else:
                query = f"""
                       MATCH (n)-[r*1..{depth}]-(m)
                       WHERE id(n) = $node_id
                       RETURN m, r;
                       """
            results = self.db.run(query, node_id=node.identity).data()

            # 遍历输出所有相邻的节点和关系
            for result in tqdm(results, desc="Fetching Nodes and Relations"):
                relationship = result['r']
                rel += str(relationship) + "\n"
            time.sleep(0.1)
            print("Nodes and Relations Fetched")
            return rel
        else:
            return None

    def create_nodes(self, entities, kg_debug):
        # 实现创建节点的逻辑
        for entity in tqdm(entities, desc="Creating Nodes"):
            if not self.matcher.match(entity["label"], name=entity["entity"]).first():
                node_tmp = Node(entity["label"], name=entity["entity"])
                if kg_debug:
                    print(f"Creating node: {entity}")
                self.db.create(node_tmp)
        print("All nodes created")

    def create_relationships(self, entities, knowledge, kg_debug):
        # 实现创建关系的逻辑
        en_list = entities
        core = en_list[0]
        core_n = core["entity"]
        sub = en_list.pop(0)
        node_core = self.matcher.match(core["label"], name=core["entity"]).first()
        for en in tqdm(en_list, desc="Creating Relationships"):
            en_n = en["entity"]
            if kg_debug:
                print(f"core: {core}, sub: {en}")
            prompt = TextProcessor.extract_surrounding_text(knowledge, en_n,
                                                            150) + f"\n根据以上信息，说明二者的关系，不需要任何其他话语和提示，请只将关系总结成一个词语,并在两边用'[]'括起来,如果二者表达同一个意思他们关系就是相同， {core_n}和{en_n} 的关系是：\n"
            res = self.agent.mode_chat(prompt)
            if kg_debug:
                print("original response: " + res)

            index_f = res.rfind('[')
            index_s = res.rfind(']')
            if index_f != -1 and index_s != -1:
                res = res[index_f + 1:index_s]
            else:
                if kg_debug:
                    print("sign not found")
            rep = {
                "\n": "",
                "<bos>": "",
                "<end_of_turn>": "",
                "*": "",
                "关键词：": "",
                " ": "",
                "答案": "",
                "：": "",
                ":": ""
            }
            pattern = re.compile("|".join(re.escape(key) for key in rep.keys()))
            rel = pattern.sub(lambda x: rep[x.group()], res)
            rel = TextProcessor.remove_repeated(rel)
            notice_index = rel.find("请注意")
            rel = (rel if notice_index == -1 else rel[:notice_index])
            notice_index = rel.find("解释")
            rel = (rel if notice_index == -1 else rel[:notice_index])
            if rel == "":
                if kg_debug:
                    print("No vivid relation")
                rel = "相关"
            if kg_debug:
                print("rel: " + rel)
            node_sub = self.matcher.match(en["label"], name=en["entity"]).first()
            core_to_sub = Relationship(node_core, rel, node_sub)
            self.db.create(core_to_sub)
        print("All relationships created")

    def kg_q_and_a(self, query, kg_debug):
        core = TextProcessor.format_entity(self.agent.extract_entities(query))

        if kg_debug:
            print("core: " + str(core))
        while True:
            # outs = []
            relations = self.rel_query(core, 0)
            if kg_debug:
                print("relations: ", end="")
                print(relations)
            if not relations:
                text = WikipediaHandler.wiki(core[0]["entity"])[:1000]
                if kg_debug:
                    print("wiki_text: " + text)
                outs = self.ner.named_entity_reco(text, kg_debug)
                if kg_debug:
                    print("outs: ", outs)
                result = outs
                time.sleep(0.1)
                self.create_nodes(result, kg_debug)
                time.sleep(0.1)
                self.create_relationships(result, text, kg_debug)
            else:
                rela = TextProcessor.format_relations(relations)
                chunks = [rela[i:i + 10] for i in range(0, len(rela), 10)]
                for chunk in chunks:
                    if kg_debug:
                        print("chunk: ", chunk)
                    prompt_qu = "你好Gemma，接下来请站在一个问答助手的角度，我将给你一些信息，请简洁明了地回答我提出的问题。"
                    inp = f"\n我给的参考信息是：{','.join(chunk)}\n请根据我给出的参考信息以及你自身知识储备回答问题，用自己的话写成一段话。不要有任何提示信息，不要分行作答，请使用陈述句回答：\n{query}\n如果无法得出答案，请不要输出。"
                    res = self.agent.mode_extended(prompt_qu, inp)
                    if kg_debug:
                        print("original response: " + res)
                    rep = {
                        prompt_qu: "",
                        inp: "",
                        "\n": "",
                        "<bos>": "",
                        "<end_of_turn>": "",
                        "*": "",
                        " ": ""
                    }
                    pattern = re.compile("|".join(re.escape(key) for key in rep.keys()))
                    outs = pattern.sub(lambda x: rep[x.group()], res)
                    if outs != "":
                        break
                else:
                    outs = "很抱歉，我没有找到相关信息"
                print("根据知识图谱，我的回答如下：")
                print("\033[33m" + outs + "\033[0m")
                print(
                    "\033[0m\n\033[32m================================================================\n\t\t\t\t  请注意，我也很可能会犯错，请小心识别\n================================================================\033[0m")
                break


class NER:
    def __init__(self, model_path):
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)

    def named_entity_reco(self, text, kg_debug):
        pieces = TextProcessor.split_text(text)
        en = []
        if kg_debug:
            print("Pieces length: ", len(pieces))
        for piece in pieces:
            encoded_input = self.tokenizer(piece, return_tensors="pt", padding=True, truncation=True)

            # 移动到设备（GPU 或 CPU）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            input_ids = encoded_input["input_ids"].to(device)
            attention_mask = encoded_input["attention_mask"].to(device)

            # 获取模型输出
            with torch.no_grad():
                output = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # 获取logits并转换为标签概率
            logits = output.logits
            probs = F.softmax(logits, dim=-1)

            # 获取每个token的预测标签
            predictions = torch.argmax(probs, dim=-1)

            # 解码标签
            predicted_labels = predictions[0].cpu().numpy()
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
            label_map = self.model.config.id2label

            # 将预测结果映射回标签并合并实体
            ner_results = []
            current_entity = []
            current_label = None

            for token, label_id in zip(tokens, predicted_labels):
                label = label_map[label_id]

                if label.startswith("B-"):  # 开始一个新的实体
                    if current_entity:
                        ner_results.append((current_entity, current_label))
                    current_entity = [token]
                    current_label = label[2:]  # 移除 "B-" 前缀

                elif label.startswith("I-") and current_label == label[2:]:  # 继续当前实体
                    current_entity.append(token)

                elif label.startswith("E-") and current_label == label[2:]:  # 结束当前实体
                    current_entity.append(token)
                    ner_results.append((current_entity, current_label))
                    current_entity = []
                    current_label = None

                elif label.startswith("S-"):  # 单字实体
                    if current_entity:
                        ner_results.append((current_entity, current_label))
                    ner_results.append(([token], label[2:]))
                    current_entity = []
                    current_label = None

            # 如果当前还有未添加的实体，将其加入结果
            if current_entity:
                ner_results.append((current_entity, current_label))

            # NER结果
            entities = []

            for entity, label in ner_results:
                if label != "DATE" and label != "CARDINAL" and label != "PERCENT" and label != "LANGUAGE" and len(
                        entity) > 1 and label != "ORDINAL" and label != "TIME":
                    entity_name = "".join(entity)
                    tmp = {"entity": TextProcessor.convert_to_simplified(entity_name), "label": label}
                    entities.append(tmp)

            en = en + entities
        field = 'entity'
        # 用于存储不重复的字典
        res = []
        seen_values = []

        for item in en:
            value = item[field]
            # 检查当前字符串是否与已有字符串存在子字符串关系
            if not any(value in seen or seen in value for seen in seen_values):
                res.append(item)
                seen_values.append(value)
        print("Recognized Entities: ", res)
        return res


class Agent:
    def __init__(self, model_path, lora_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(self.model, lora_path)
        self.prompt_kw = '''
你是一位助手，现在需要根据给出的问题生成一段JSON格式的文本。这个文本需要包含以下内容：
query: 将给定的问题直接放在 "query" 字段中。
sequence: 生成一系列关键字或短语，这些内容是回答该问题时需要查询的主要信息。每个关键字或短语之间使用 [des] 作为分隔符。注意，sequence 中的内容要紧密围绕问题，并且每一个关键字或短语都要有独立的意义，可以单独查询。
请按照以下格式生成输出：

json
{"query": "给定的问题", "sequence": "关键字1[des]关键字2[des]关键字3..."}
示例：

问题：如何策划一场完美的婚礼？

{"query": "如何策划一场完美的婚礼？", "sequence": "选择婚礼场地的要点[des]婚礼主题与风格设计[des]婚礼策划公司的选择[des]婚礼当天的流程安排[des]婚礼摄影与录像服务"}
问题：如何在家中种植蔬菜？

{"query": "如何在家中种植蔬菜？", "sequence": "适合室内种植的蔬菜品种[des]家庭种菜的土壤选择[des]室内种植的光照与浇水需求[des]如何预防蔬菜的病虫害[des]种植蔬菜的收获时间"}
请根据上面的格式生成对应的JSON文本。
    '''
        self.prompt_net_res = "你好Gemma，接下来请站在一个联网问答助手的角度，我将给你一些网络上的信息，请简介明了地回答我提出的问题。"
        self.prompt_ner = '''任务: 在下面的段落中识别出所有的人名（Person）、组织名（Organization）和地名（Location）实体。输出格式为JSON，每个实体包括实体类别和文本。

    输入: "李雷在北京大学学习，后来加入了谷歌。"

    输出:
    [
        {"entity": "李雷", "label": "Person"},
        {"entity": "北京大学", "label": "Organization"},
        {"entity": "谷歌", "label": "Organization"}
    ]

    现在，请在下面的句子中识别所有的实体,不要有任何提示词，直接按照格式输出json列表表示出所有的实体，实体分类只有Person，Organization，Location,Identity,Country,Date和Other这几类，请注意：一定不要重复提取同一个实体。
    请注意：请直接提取实体输出，不需要任何代码演示。

    句子: '''

    def mode_extended(self, prompt, inputs=""):
        input_text = prompt + inputs
        input_ids = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = self.model.generate(**input_ids, max_new_tokens=1000)
        result = self.tokenizer.decode(outputs[0])
        return result

    def mode_chat(self, prompt, inputs=""):
        messages = [
            {"role": "user", "content": prompt + inputs},
        ]
        input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")
        outputs = self.model.generate(**input_ids, max_new_tokens=512)
        return self.tokenizer.decode(outputs[0])

    def extract_entities(self, text):
        # 实现实体提取的逻辑
        return self.mode_chat(self.prompt_ner, text).replace(self.prompt_ner, "")

    def keywords_extractor(self, query):
        result = self.mode_extended(self.prompt_kw, query)
        matches = re.findall(r'"(.*?)"', result)
        if matches:
            # 取出最后一个匹配项
            last_match = matches[-1]
        else:
            last_match = None
            print("未找到双引号括起来的内容")
        # 使用 '[des]' 作为分隔符进行分割
        components = last_match.split('[des]')
        # 去除列表中的空字符串
        components = [comp for comp in components if comp]
        print(components)
        return components

    def generate_response_net(self, query, kn, debug):
        inputs = f"\n我给的参考信息是：{str(kn)}\n请根据我给出的参考信息以及你自身知识储备回答问题，请必须分点作答，每一点用自己的话写成一段话。不要有任何提示信息，请回答我的问题不要提出问题。：\n{query}\n"
        if debug:
            print(self.mode_chat(self.prompt_net_res, inputs))
        res = TextProcessor.format_an(self.mode_chat(self.prompt_net_res, inputs)).strip()
        res = re.sub("\n+", "\n", res)
        print(
            "搜索结果显示：\n" + "\033[33m" + res + "\033[0m\n\033[32m================================================================\n\t\t\t\t请注意，所有信息均来自网络，请小心识别\n================================================================\033[0m")


class Summarizer:
    def __init__(self, model_path):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to("cuda")
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

    def T5_sum(self, text, debug=False):
        if text.strip() != "" or None:
            if debug:
                print("text: ", text)
            prefix = 'summary big:'
            src_text = prefix + text[:512]
            input_ids = self.tokenizer(src_text, return_tensors="pt").to("cuda")

            generated_tokens = self.model.generate(**input_ids)

            result = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            return "".join(result)
        else:
            return ""


class WebSearcher:
    def __init__(self, summarizer, firecrawl, bing_api_key):
        self.summarizer = summarizer
        self.knowledge = {}
        self.firecrawl = firecrawl
        self.bing_api_key = bing_api_key

    def Baidubaike(self, item, debug):
        url = 'https://baike.baidu.com/item/' + urllib.parse.quote(item)
        headers = {"user-agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 "
                                 "(KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"
                   }
        # 利用请求地址和请求头部构造请求对象
        req = urllib.request.Request(url=url, headers=headers, method='GET')
        # 发送请求，获得响应
        response = urllib.request.urlopen(req)
        # 读取响应，获得文本
        text = response.read().decode('utf-8')
        # 构造 _Element 对象
        html = etree.HTML(text)
        # 使用 xpath 匹配数据，得到匹配字符串列表
        sen_list = html.xpath('//div[contains(@class,"J-lemma-content") ]//text()')
        # 过滤数据，去掉空白
        sen_list_after_filter = [item.strip('\n') for item in sen_list]
        # 将字符串列表连成字符串并返回
        result = ''.join(sen_list_after_filter)
        # print(result)
        if result.strip() != "":
            return self.summarizer.T5_sum(TextProcessor.Ran_sele(result.strip()), debug)
        else:
            return ""

    def search(self, w, debug):
        self.knowledge = {}
        for query in w:
            global ctent
            ctent = ""
            # 获取搜索结果的第一个网址
            subscription_key = self.bing_api_key
            search_url = "https://api.bing.microsoft.com/v7.0/search"
            headers = {"Ocp-Apim-Subscription-Key": subscription_key}
            params = {"q": query, "count": 10}

            response = requests.get(search_url, headers=headers, params=params)
            results = response.json()

            # 提取n个结果的网址
            result_url = results['webPages']['value'][1:5]

            proxies = {
                "http": "http://127.0.0.1:7890",
                "https": "http://127.0.0.1:7890",
            }
            headers = {"user-agent": "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 "
                                     "(KHTML, like Gecko) Chrome/80.0.3987.132 Safari/537.36"
                       }

            # 抓取网页内容
            def get(url):
                article_text = ""
                response = requests.get(url, proxies=proxies, headers=headers, verify=False)
                html_content = response.text

                # 解析网页内容
                soup = BeautifulSoup(html_content, 'html.parser')

                # 示例：提取网页中的标题和第一段文本
                title = soup.title.string if soup.title else 'No title'
                if title.endswith("百度百科"):
                    b_index = title.find('（')
                    if b_index != -1:
                        # 从左括号之前的部分创建新字符串
                        item = title[:b_index]
                    else:
                        # 如果没有找到左括号，返回原始字符串
                        item = title.replace("_百度百科", "")
                    article_text = self.Baidubaike(item, debug)

                else:
                    article_body = self.firecrawl.get_text_from_url(url)

                    # 如果找到正文内容，则提取其文本
                    if article_body:
                        article_text = re.sub(r'\s+', ' ', article_body).strip()
                    else:
                        article_text = ''

                    # 清理多余的换行和空白
                    # 替换多个换行符为一个空格，然后去除首尾的换行符
                    article_text = re.sub(r'\n+', ' ', article_text).strip()
                    # 打印结果

                    if debug:
                        print(f"Title: {title}")
                        print(f"Article Content: {article_text}")  # 打印出来
                        print(f"Url: {url}")

                if article_text.strip() != "" or " ":
                    return self.summarizer.T5_sum(
                        TextProcessor.Ran_sele("标题：" + title + "文章内容：" + article_text.strip()), debug)
                else:
                    return self.summarizer.T5_sum(TextProcessor.Ran_sele("标题：" + title), debug)

            for U in tqdm(result_url, desc="Fetching search results"):
                try:
                    tmp = get(U['url'])
                    ctent += "\n" + tmp if tmp != "" else ""
                except Exception as e:
                    print(f"出现错误：{e}")
                    continue
            if debug:
                print("================================================================")
                print(ctent)
                print("\n================================================================")
            print(f"Knownledge: {query} fetched")
            self.knowledge[query] = ctent.replace("段落内容主要涉及到以下几个主题:", "")
        return self.knowledge


class ChatBot:
    def __init__(self, agent, knowledge_graph, summarizer, web_searcher):
        self.query = ""
        self.kg_debug = False
        self.net_debug = False
        self.agent = agent
        self.knowledge_graph = knowledge_graph
        self.summarizer = summarizer
        self.web_searcher = web_searcher
        self.know_net = {}
        self.know_kg = {}

    def run(self):
        while True:
            print(
                "\033[0m\n\033[34m================================================================\n\t\t\t请选择： 1.联网查询   2.知识图谱问答   0.退出系统\n================================================================\033[0m")
            choice = input("请输入：")
            if choice == "1":
                try:
                    tmp = input("现在是联网查询，请输入问题：").split("*", 1)
                    if len(tmp) > 1 and tmp[1] == "debug":
                        self.net_debug = True
                    self.query = tmp[0]
                    k_words = self.agent.keywords_extractor(self.query)
                    self.know_net = self.web_searcher.search(k_words, self.net_debug)
                    if self.net_debug:
                        print(self.know_net)
                    self.agent.generate_response_net(self.query, self.know_net, self.net_debug)
                except Exception as e:
                    print(f"出现错误：{e}")
                    continue
            elif choice == "2":
                tmp = input("现在是知识图谱问答，请输入问题：").split("*", 1)
                if len(tmp) > 1 and tmp[1] == "debug":
                    self.kg_debug = True
                if self.kg_debug:
                    print("query: " + tmp[0])
                self.knowledge_graph.kg_q_and_a(tmp[0], self.kg_debug)
            elif choice == "0":
                return
            else:
                print("\033[31m\n\t\t\t输入有误，请重新输入\033[0m")
                continue


# 主程序
if __name__ == "__main__":
    warnings.simplefilter('ignore', InsecureRequestWarning)
    agent = Agent("model/gemma-2-2b-it", "model/Gemma_lora_model")
    ner = NER("model/bert-base-chinese-ner")
    knowledge_graph = KnowledgeGraph(ner, agent, 'bolt://localhost:7687', auth=("name", "password"))
    summarizer = Summarizer("model/t5_summary")
    firecrawl = Firecrawl(api_key='API_KEY')
    web_searcher = WebSearcher(summarizer, firecrawl, bing_api_key='API_KEY')
    chatbot = ChatBot(agent, knowledge_graph, summarizer, web_searcher)
    chatbot.run()
