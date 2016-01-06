

from Utils_Base import *
import os

# REQ

import requests as _REQ

class REQ_C():
	
	def get(self,*args,**kwargs):
		return _REQ.get(*args,**kwargs)
		
# REQ=REQ_C()

# Selenium

from selenium import webdriver as _webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys

def webdriver(core_name):
	core_name=core_name.lower()
	if core_name=="firefox":
		browser=_webdriver.Firefox()
	elif core_name=="phantomjs":
		if "win" in sys.platform:
			browser=_webdriver.PhantomJS(
			executable_path=PATH_JOIN(
				CUR_PATH,'_webdriver/phantomjs.exe'))
		elif "linux" in sys.platform:
			browser=_webdriver.PhantomJS()
	
	return Browser(browser)
	
class Browser():
	
	def __init__(self,browser, webdriver=None):
		self.browser=browser
		self.webdriver=webdriver
		
	def get(self,*args,**kwargs):
		self.browser.get(*args,**kwargs)
		return self
		
	def find_element_by_xpath(self,*args,**kwargs):
		return self.browser.find_element_by_xpath(*args,**kwargs)
		
	def scroll_to_end(self):
		memory={'last_scrollHeight':0,'cur_scrollHeight':0,'counter':0}
		
		def do_scroll(browser):
			browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")
			cur_scrollHeight=browser.execute_script("return document.body.scrollHeight;")
			return cur_scrollHeight
			
		def scroll_until(cur_scrollHeight,memory):
			if cur_scrollHeight>memory['cur_scrollHeight']:
				flag=False
			else:
				if memory['counter']==3:
					memory['counter']=0
					flag=True
				else:
					memory['counter']+=1
					time.sleep(1)
					flag=False
				
			memory['cur_scrollHeight']=cur_scrollHeight
			return flag
		
		do_until(do_scroll,scroll_until,do_params=self.browser,until_params=memory)
		
		return self

	def next_page(self):
		pass

	def close(self):
		self.browser.close()
	
	def __getattr__(self,key):
		if key=="page_source":
			return self.browser.page_source
		else:
			return getattr(self.browser, key)

# BSplus

from bs4 import BeautifulSoup as BS

class BSplus_C():

	def __init__(self,page,parser):
		self.parser=parser
		self.page=page
	
	def parse(self):
		result=parse_attr(self.parser.parser_tree,self.page)
		for key,val in result.items():
			func=getattr(self.parser,"P_"+key,None)
			if not func==None:
				result[key]=func(val)
		
		return result
	
def parse_attr(parser_tree,page):
	parser_tree=BS(parser_tree)
	bs=BS(page)
	
	if not parser_tree.find('root')==None:
		parser_tree=parser_tree.root
	
	attr_tag_dict={}
	attrs=parser_tree.find_all("attr")
	for attr in attrs:
		name=attr['name']
		all=attr.get("all")
		
		for tag in attr.find_all(True,recrusive=False):
			result=parse_tag(bs,tag)
			if not len(result)==0:
				if not all:
					result=result[0]
				attr_tag_dict[name]=result
				break
	
	return attr_tag_dict
	
def parse_tag(bs,tag):

	"""
	nodes=find_all_node(tag.cond)
	if not len(tag.content)==0:
		targets_nodes=[]
		for node in nodes:
			for ctag in tag.children:
				result=parse(node,ctag)
				targets_nodes.extend(result)
	
	else:
		targets_nodes=nodes
		
	return targets_nodes
	"""

	# tag cond
	attrs=tag.attrs.copy()
	# print attrs
	
	if "class" in attrs.keys():
		attrs['class_']=' '.join(attrs['class'])
	
	cmd_dict={}
	for key in ["has_string"]:
		if key in attrs.keys():
			cmd_dict[key]=attrs[key]
			del attrs[key]

	# find all node, if none should return []
	nodes=bs.find_all(tag.name,**attrs)
	
	# run cmd has_string
	if "has_string" in cmd_dict.keys():
		temp_dict={}
		for s in cmd_dict['has_string'].split(";")[:-1]:
			attrname,attrval=s.split(":")
			temp_dict[attrname]=attrval
		
		_nodes=[]
		for i,node in enumerate(nodes):
			ok=True
			for key,val in temp_dict.items():
				try:
					s=node[key]					
					if not val in s:
						ok=False
						break
				except KeyError:
					pass
			
			if ok:
				_nodes.append(node)
		
		nodes=_nodes
	
	# gen target_nodes
	# print tag
	if not len(tag.find_all(True,recursive=False))==0:
		target_nodes=[]
		for node in nodes:
			for ctag in tag.find_all(True,recursive=False):
				result=parse_tag(node,ctag)
				target_nodes.extend(result)
	else:
		target_nodes=nodes
		
	# try:
		# if "zm-profile-side-following" in tag['class']:
			# print nodes
	# except:
		# pass
		
	return target_nodes
	
BSplus=BSplus_C

