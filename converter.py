import json

def separator(cadena):
  if(type(cadena) != str):
      return "it is not a string"

  dic = json.loads(cadena)
  return dic
