# langchain document loader for CodeChatBot

import os
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import WebBaseLoader
import requests

class DocumentLoader:
    def __init__(self):
        pass

    def load_csv(self, file_path):
        """
        Load a csv file
        :param file_path:
        :return:
        """
        loader = CSVLoader(file_path)
        return loader.load()


    def load_pdf(self, file_path):
        """
        Load a pdf file.
        :param file_path:
        :return: clean_text, length
        """
        loader = PyPDFLoader(file_path)
        clean_text = dict(loader.load()[0])['page_content']
        # remove all unnecessary characters from the text
        clean_text = clean_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\x0c', ' ').replace('•', ' ')
        length = len(clean_text)
        return clean_text, length


    def load_text(self, file_path):
        """
        Load a text file
        :param file_path:
        :return: clean_text, length
        """
        loader = TextLoader(file_path)
        clean_text = dict(loader.load()[0])['page_content']
        # remove all unnecessary characters from the text
        clean_text = clean_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\x0c', ' ').replace('•',
                                                                                                                      ' ')
        length = len(clean_text)
        return clean_text, length


    def load_web(self, url):
        """
        Load a web page
        :param url:
        :return: clean_text, length
        """
        loader = WebBaseLoader(url)
        clean_text = dict(loader.load()[0])['page_content']
        # remove all unnecessary characters from the text
        clean_text = clean_text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\x0c', ' ').replace('•',
                                                                                                                      ' ')
        clean_text = ' '.join(clean_text.split())
        length = len(clean_text)
        return clean_text, length

first = DocumentLoader()

print(first.load_web('https://www.w3schools.com/python/python_intro.asp'))