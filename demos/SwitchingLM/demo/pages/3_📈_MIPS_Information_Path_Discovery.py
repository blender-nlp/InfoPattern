import streamlit as st
import graphviz
import json
import os

st.markdown('## 📈 Information Propagation Path Discovery')

with st.expander("ℹ️ - Model Architecture (Path Language Model)", expanded=False):
        st.write(
            """     
    -   This page shows the information propagation paths we discovered from Twitter.
    -   It uses a path language model to learn the generalization patterns for each theme. 
            """
        )
        st.markdown("")
        st.image("infopath.png", width=800)

# # Create a graphlib graph object
# graph_name = "Theme 1 Philippine Politics"
# graph = graphviz.Digraph(graph_name, comment='test graph', format='png', node_attr={'shape': 'rectangle', 'color': 'lightblue2', 'style':"rounded,filled", 'fontsize': "16"})
# graph.edge('run', 'intr')
# graph.edge('intr', 'runbl')
# graph.edge('runbl', 'run')
# graph.edge('run', 'kernel')
# graph.edge('kernel', 'zombie')
# graph.edge('kernel', 'sleep')
# graph.edge('kernel', 'runmem')
# graph.edge('sleep', 'swap')
# graph.edge('swap', 'runswap')
# graph.edge('runswap', 'new')
# graph.edge('runswap', 'runmem')
# graph.edge('new', 'runmem')
# graph.edge('sleep', 'runmem')

# st.graphviz_chart(graph)

for jsonfile in os.listdir("data"):
  if jsonfile.endswith('.json'):
    # load data
    data = json.load(open('data/%s' % jsonfile))
    with st.expander(data["title"], expanded=True):
      # Define your javascript
      my_js = """
      // Initialize the echarts instance based on the prepared dom
      var myChart = echarts.init(document.getElementById('main'));
      var option = {
        title: {
          text: '%s'
        },
        tooltip: {},
        animationDurationUpdate: 1500,
        animationEasingUpdate: 'quinticInOut',
        series: [
          {
            type: 'graph',
            layout: 'none',
            symbolSize: 50,
            roam: true,
            legend: [
              {
                data: ["Left", "Neutral", "Right"]
              }
            ],
            label: {
              normal: {
                  formatter: '{b}',
                  show: true
              },
            },
            edgeSymbol: ['circle', 'arrow'],
            edgeSymbolSize: [4, 10],
            edgeLabel: {
              fontSize: 20
            },
            categories: %s,
            data: %s,
            edges: %s,
            lineStyle: {
              opacity: 0.9,
              width: 2,
              curveness: 0.1
            },
            legendHoverLink: true,
            draggable: true,
            focusNodeAdjacency: true,
            force: {
                // edgeLength: 100,
                // repulsion: 100,
                // gravity: 0.2
                edgeLength: 100,
                repulsion: 200,
                gravity: 0.03
            }
          }
        ]
      };

      // Display the chart using the configuration items and data just specified.
      myChart.setOption(option);
      """ % (data["title"], data["categories"], data["nodes"], data["links"])

      # Wrapt the javascript as html code
      my_html = f"""
      <!DOCTYPE html>
      <html>
        <head>
          <!-- Include the ECharts file you just downloaded -->
              <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts/dist/echarts.min.js"></script>
              <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts-gl/dist/echarts-gl.min.js"></script>
              <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts-stat/dist/ecStat.min.js"></script>
              <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts/dist/extension/dataTool.min.js"></script>
              <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts/map/js/china.js"></script>
              <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts/map/js/world.js"></script>
              <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/echarts/dist/extension/bmap.min.js"></script>
              <script src="https://cdn.bootcss.com/jquery/3.2.1/jquery.min.js"></script>
        </head>
        <body>
          <!-- Prepare a DOM with a defined width and height for ECharts -->
          <div id="main" style="width: 900px;height: 600px;"></div>
          <script type="text/javascript">
            {my_js}
          </script>
        </body>
      </html>
        """
      
      st.components.v1.html(
          my_html,
          height=600
      )