import pandas as pd
import xml.dom.minidom 
pd.options.mode.chained_assignment = None
#from xml import Node

df = pd.read_csv('/Users/sukshi/Downloads/ab.csv')
def convert_row(row):
    print(row)
    return """<xcall_id="%s">
    <landmarks_values>%s</landmarks_values>
    </xcall_id>""" % (row.xcall_id, row.landmarks_values)


# print('\n'.join(df.apply(convert_row, axis=1)))
c= '\n'.join(df.apply(convert_row,axis=1))
# print(c)
file=open("shrus.xml","w")
#n = Node.writexml(writer, indent="convert_row", addindent="", newl="")
file.write(c)