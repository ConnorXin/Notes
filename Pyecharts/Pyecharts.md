```python
# 使用 pyecharts 必须导入以下包
from pyecharts import options as opts
fig.render_notebook()
```

# 全局配置项

[全局配置项 - pyecharts - A Python Echarts Plotting Library built with love.](https://pyecharts.org/#/zh-cn/global_options)

## TitleOpts - 标题

`set_global_opts` 非常重要的修饰函数，负责图形大部分整体的修饰

```python
class TitleOpts(
    # 主标题文本，支持使用 \n 换行。
    title: Optional[str] = None,

    # 主标题跳转 URL 链接
    title_link: Optional[str] = None,

    # 主标题跳转链接方式
    # 默认值是: blank
    # 可选参数: 'self', 'blank'
    # 'self' 当前窗口打开; 'blank' 新窗口打开
    title_target: Optional[str] = None,

    # 副标题文本，支持使用 \n 换行。
    subtitle: Optional[str] = None,

    # 副标题跳转 URL 链接
    subtitle_link: Optional[str] = None,

    # 副标题跳转链接方式
    # 默认值是: blank
    # 可选参数: 'self', 'blank'
    # 'self' 当前窗口打开; 'blank' 新窗口打开
    subtitle_target: Optional[str] = None,

    # title 组件离容器左侧的距离。
    # left 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比，
    # 也可以是 'left', 'center', 'right'。
    # 如果 left 的值为'left', 'center', 'right'，组件会根据相应的位置自动对齐。
    pos_left: Optional[str] = None,

    # title 组件离容器右侧的距离。
    # right 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比。
    pos_right: Optional[str] = None,

    # title 组件离容器上侧的距离。
    # top 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比，
    # 也可以是 'top', 'middle', 'bottom'。
    # 如果 top 的值为'top', 'middle', 'bottom'，组件会根据相应的位置自动对齐。
    pos_top: Optional[str] = None,

    # title 组件离容器下侧的距离。
    # bottom 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比。
    pos_bottom: Optional[str] = None,

    # 标题内边距，单位px，默认各方向内边距为5，接受数组分别设定上右下左边距。
    # // 设置内边距为 5
    # padding: 5
    # // 设置上下的内边距为 5，左右的内边距为 10
    # padding: [5, 10]
    # // 分别设置四个方向的内边距
    # padding: [
    #     5,  // 上
    #     10, // 右
    #     5,  // 下
    #     10, // 左
    # ]
    padding: Union[Sequence, Numeric] = 5,

    # 主副标题之间的间距。
    item_gap: Numeric = 10,

    # 主标题字体样式配置项，参考 `series_options.TextStyleOpts`
    title_textstyle_opts: Union[TextStyleOpts, dict, None] = None,

    # 副标题字体样式配置项，参考 `series_options.TextStyleOpts`
    subtitle_textstyle_opts: Union[TextStyleOpts, dict, None] = None,
)
```

### e.g.

```python
bar = Bar()
bar.add_xaxis(list(x))
bar.add_yaxis('name', y)
bar.set_global_opts(title_opts = opts.TitleOpts(title = 'Bar-渐变圆柱'))
```

## AxisOpts - 坐标轴

```python
class AxisOpts(
    # 坐标轴类型。可选：
    # 'value': 数值轴，适用于连续数据。
    # 'category': 类目轴，适用于离散的类目数据，为该类型时必须通过 data 设置类目数据。
    # 'time': 时间轴，适用于连续的时序数据，与数值轴相比时间轴带有时间的格式化，在刻度计算上也有所不同，
    # 例如会根据跨度的范围来决定使用月，星期，日还是小时范围的刻度。
    # 'log' 对数轴。适用于对数数据。
    type_: Optional[str] = None,

    # 坐标轴名称。
    name: Optional[str] = None,

    # 是否显示 x 轴。
    is_show: bool = True,

    # 只在数值轴中（type: 'value'）有效。
    # 是否是脱离 0 值比例。设置成 true 后坐标刻度不会强制包含零刻度。在双数值轴的散点图中比较有用。
    # 在设置 min 和 max 之后该配置项无效。
    is_scale: bool = False,

    # 是否反向坐标轴。
    is_inverse: bool = False,

    # 坐标轴名称显示位置。可选：
    # 'start', 'middle' 或者 'center','end'
    name_location: str = "end",

    # 坐标轴名称与轴线之间的距离。
    name_gap: Numeric = 15,

    # 坐标轴名字旋转，角度值。
    name_rotate: Optional[Numeric] = None,

    # 强制设置坐标轴分割间隔。
    # 因为 splitNumber 是预估的值，实际根据策略计算出来的刻度可能无法达到想要的效果，
    # 这时候可以使用 interval 配合 min、max 强制设定刻度划分，一般不建议使用。
    # 无法在类目轴中使用。在时间轴（type: 'time'）中需要传时间戳，在对数轴（type: 'log'）中需要传指数值。
    interval: Optional[Numeric] = None,

    # x 轴所在的 grid 的索引，默认位于第一个 grid。
    grid_index: Numeric = 0,

    # x 轴的位置。可选：
    # 'top', 'bottom'
    # 默认 grid 中的第一个 x 轴在 grid 的下方（'bottom'），第二个 x 轴视第一个 x 轴的位置放在另一侧。
    position: Optional[str] = None,

    # Y 轴相对于默认位置的偏移，在相同的 position 上有多个 Y 轴的时候有用。
    offset: Numeric = 0,

    # 坐标轴的分割段数，需要注意的是这个分割段数只是个预估值，最后实际显示的段数会在这个基础上根据分割后坐标轴刻度显示的易读程度作调整。 
    # 默认值是 5
    split_number: Numeric = 5,

    # 坐标轴两边留白策略，类目轴和非类目轴的设置和表现不一样。
    # 类目轴中 boundaryGap 可以配置为 true 和 false。默认为 true，这时候刻度只是作为分隔线，
    # 标签和数据点都会在两个刻度之间的带(band)中间。
    # 非类目轴，包括时间，数值，对数轴，boundaryGap是一个两个值的数组，分别表示数据最小值和最大值的延伸范围
    # 可以直接设置数值或者相对的百分比，在设置 min 和 max 后无效。 示例：boundaryGap: ['20%', '20%']
    boundary_gap: Union[str, bool, None] = None,

    # 坐标轴刻度最小值。
    # 可以设置成特殊值 'dataMin'，此时取数据在该轴上的最小值作为最小刻度。
    # 不设置时会自动计算最小值保证坐标轴刻度的均匀分布。
    # 在类目轴中，也可以设置为类目的序数（如类目轴 data: ['类A', '类B', '类C'] 中，序数 2 表示 '类C'。
    # 也可以设置为负数，如 -3）。
    min_: Union[Numeric, str, None] = None,

    # 坐标轴刻度最大值。
    # 可以设置成特殊值 'dataMax'，此时取数据在该轴上的最大值作为最大刻度。
    # 不设置时会自动计算最大值保证坐标轴刻度的均匀分布。
    # 在类目轴中，也可以设置为类目的序数（如类目轴 data: ['类A', '类B', '类C'] 中，序数 2 表示 '类C'。
    # 也可以设置为负数，如 -3）。
    max_: Union[Numeric, str, None] = None,

    # 自动计算的坐标轴最小间隔大小。
    # 例如可以设置成1保证坐标轴分割刻度显示成整数。
    # 默认值是 0
    min_interval: Numeric = 0,

    # 自动计算的坐标轴最大间隔大小。
    # 例如，在时间轴（（type: 'time'））可以设置成 3600 * 24 * 1000 保证坐标轴分割刻度最大为一天。
    max_interval: Optional[Numeric] = None,

    # 坐标轴刻度线配置项，参考 `global_options.AxisLineOpts`
    '''类中调用其他类，继承关系，详细举例见 e.g.1 '''
    axisline_opts: Union[AxisLineOpts, dict, None] = None,

    # 坐标轴刻度配置项，参考 `global_options.AxisTickOpts`
    axistick_opts: Union[AxisTickOpts, dict, None] = None,

    # 坐标轴标签配置项，参考 `series_options.LabelOpts`
    axislabel_opts: Union[LabelOpts, dict, None] = None,

    # 坐标轴指示器配置项，参考 `global_options.AxisPointerOpts`
    axispointer_opts: Union[AxisPointerOpts, dict, None] = None,

    # 坐标轴名称的文字样式，参考 `series_options.TextStyleOpts`
    name_textstyle_opts: Union[TextStyleOpts, dict, None] = None,

    # 分割区域配置项，参考 `series_options.SplitAreaOpts`
    splitarea_opts: Union[SplitAreaOpts, dict, None] = None,

    # 分割线配置项，参考 `series_options.SplitLineOpts`
    splitline_opts: Union[SplitLineOpts, dict] = SplitLineOpts(),

    # 坐标轴次刻度线相关设置，参考 `series_options.MinorTickOpts`
    minor_tick_opts: Union[MinorTickOpts, dict, None] = None,

    # 坐标轴在 grid 区域中的次分隔线。次分割线会对齐次刻度线 minorTick，参考 `series_options.MinorSplitLineOpts`
    minor_split_line_opts: Union[MinorSplitLineOpts, dict, None] = None,
)
```

### e.g.

```python
bar = Bar()
bar.set_global_opts(xaxis_opts = opts.AxisOpts(max_ = 6))
```

```python
'''e.g.1'''
bar = Bar()
bar.add_xaxis(list(x))
bar.add_yaxis('name', y)
bar.set_global_opts(xaxis_opts = opts.AxisOpts(max_ = 6,
                    axisline_opts = opts.AxisLineOpts(is_show = False)))
```

## AxisLineOpts - 坐标轴轴线

```python
class AxisLineOpts(
    # 是否显示坐标轴轴线。
    is_show: bool = True,

    # X 轴或者 Y 轴的轴线是否在另一个轴的 0 刻度上，只有在另一个轴为数值轴且包含 0 刻度时有效。
    is_on_zero: bool = True,

    # 当有双轴时，可以用这个属性手动指定，在哪个轴的 0 刻度上。
    on_zero_axis_index: int = 0,

    # 轴线两边的箭头。可以是字符串，表示两端使用同样的箭头；或者长度为 2 的字符串数组，分别表示两端的箭头。
    # 默认不显示箭头，即 'none'。
    # 两端都显示箭头可以设置为 'arrow'。
    # 只在末端显示箭头可以设置为 ['none', 'arrow']。
    symbol: Optional[str] = None,

    # 坐标轴线风格配置项，参考 `series_optionsLineStyleOpts`
    linestyle_opts: Union[LineStyleOpts, dict, None] = None,
)
```

## DataZoomOpts - 区域缩放配置项

```python
class DataZoomOpts(
    # 是否显示 组件。如果设置为 false，不会显示，但是数据过滤的功能还存在。
    is_show: bool = True,

    # 组件类型，可选 "slider", "inside"
    type_: str = "slider",

    # 拖动时，是否实时更新系列的视图。如果设置为 false，则只在拖拽结束的时候更新。
    is_realtime: bool = True,

    # 数据窗口范围的起始百分比。范围是：0 ~ 100。表示 0% ~ 100%。
    range_start: Union[Numeric, None] = 20,

    # 数据窗口范围的结束百分比。范围是：0 ~ 100
    range_end: Union[Numeric, None] = 80,

    # 数据窗口范围的起始数值。如果设置了 start 则 startValue 失效。
    start_value: Union[int, str, None] = None,

    # 数据窗口范围的结束数值。如果设置了 end 则 endValue 失效。
    end_value: Union[int, str, None] = None,

    # 布局方式是横还是竖。不仅是布局方式，对于直角坐标系而言，也决定了，缺省情况控制横向数轴还是纵向数轴
    # 可选值为：'horizontal', 'vertical'
    orient: str = "horizontal",

    # 设置 dataZoom-inside 组件控制的 x 轴（即 xAxis，是直角坐标系中的概念，参见 grid）。
    # 不指定时，当 dataZoom-inside.orient 为 'horizontal'时，默认控制和 dataZoom 平行的第一个 xAxis
    # 如果是 number 表示控制一个轴，如果是 Array 表示控制多个轴。
    xaxis_index: Union[int, Sequence[int], None] = None,

    # 设置 dataZoom-inside 组件控制的 y 轴（即 yAxis，是直角坐标系中的概念，参见 grid）。
    # 不指定时，当 dataZoom-inside.orient 为 'horizontal'时，默认控制和 dataZoom 平行的第一个 yAxis
    # 如果是 number 表示控制一个轴，如果是 Array 表示控制多个轴。
    yaxis_index: Union[int, Sequence[int], None] = None,

    # 是否锁定选择区域（或叫做数据窗口）的大小。
    # 如果设置为 true 则锁定选择区域的大小，也就是说，只能平移，不能缩放。
    is_zoom_lock: bool = False,

    # dataZoom-slider 组件离容器左侧的距离。
    # left 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比，
    # 也可以是 'left', 'center', 'right'。
    # 如果 left 的值为 'left', 'center', 'right'，组件会根据相应的位置自动对齐。
    pos_left: Optional[str] = None,

    # dataZoom-slider 组件离容器上侧的距离。
    # top 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比，
    # 也可以是 'top', 'middle', 'bottom'。
    # 如果 top 的值为 'top', 'middle', 'bottom'，组件会根据相应的位置自动对齐。
    pos_top: Optional[str] = None,

    # dataZoom-slider 组件离容器右侧的距离。
    # right 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比。
    # 默认自适应。
    pos_right: Optional[str] = None,

    # dataZoom-slider组件离容器下侧的距离。
    # bottom 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比。
    # 默认自适应。
    pos_bottom: Optional[str] = None,

    # dataZoom 的运行原理是通过数据过滤以及在内部设置轴的显示窗口来达到数据窗口缩放的效果。
    # 'filter'：当前数据窗口外的数据，被过滤掉。即会影响其他轴的数据范围。
    #  每个数据项，只要有一个维度在数据窗口外，整个数据项就会被过滤掉。
    # 'weakFilter'：当前数据窗口外的数据，被过滤掉。即会影响其他轴的数据范围。
    #  每个数据项，只有当全部维度都在数据窗口同侧外部，整个数据项才会被过滤掉。
    # 'empty'：当前数据窗口外的数据，被设置为空。即不会影响其他轴的数据范围。
    # 'none': 不过滤数据，只改变数轴范围。
    filter_mode: str = "filter"
)
```

# 系列配置项

[系列配置项 - pyecharts - A Python Echarts Plotting Library built with love.](https://pyecharts.org/#/zh-cn/series_options)

`set_series_opts` 负责很多系列配置项的定义，比如 LabelOpts, MarkPointOpts, AreaStyleOpts

## LabelOpts - 标签

```python
class LabelOpts(
    # 是否显示标签。
    is_show: bool = True,

    # 标签的位置。可选
    # 'top'，'left'，'right'，'bottom'，'inside'，'insideLeft'，'insideRight'
    # 'insideTop'，'insideBottom'， 'insideTopLeft'，'insideBottomLeft'
    # 'insideTopRight'，'insideBottomRight'
    position: Union[str, Sequence] = "top",

    # 文字的颜色。
    # 如果设置为 'auto'，则为视觉映射得到的颜色，如系列色。
    color: Optional[str] = None,

    # 距离图形元素的距离。当 position 为字符描述值（如 'top'、'insideRight'）时候有效。
    distance: Union[Numeric, Sequence, None] = None,

    # 文字的字体大小
    font_size: Numeric = 12,

    # 文字字体的风格，可选：
    # 'normal'，'italic'，'oblique'
    font_style: Optional[str] = None,

    # 文字字体的粗细，可选：
    # 'normal'，'bold'，'bolder'，'lighter'
    font_weight: Optional[str] = None,

    # 文字的字体系列
    # 还可以是 'serif' , 'monospace', 'Arial', 'Courier New', 'Microsoft YaHei', ...
    font_family: Optional[str] = None,

    # 标签旋转。从 -90 度到 90 度。正值是逆时针。
    rotate: Optional[Numeric] = None,

    # 刻度标签与轴线之间的距离。
    margin: Optional[Numeric] = 8,

    # 坐标轴刻度标签的显示间隔，在类目轴中有效。
    # 默认会采用标签不重叠的策略间隔显示标签。
    # 可以设置成 0 强制显示所有标签。
    # 如果设置为 1，表示『隔一个标签显示一个标签』，如果值为 2，表示隔两个标签显示一个标签，以此类推。
    # 可以用数值表示间隔的数据，也可以通过回调函数控制。回调函数格式如下：
    # (index:number, value: string) => boolean
    # 第一个参数是类目的 index，第二个值是类目名称，如果跳过则返回 false。
    interval: Union[Numeric, str, None]= None,

    # 文字水平对齐方式，默认自动。可选：
    # 'left'，'center'，'right'
    horizontal_align: Optional[str] = None,

    # 文字垂直对齐方式，默认自动。可选：
    # 'top'，'middle'，'bottom'
    vertical_align: Optional[str] = None,

    # 标签内容格式器，支持字符串模板和回调函数两种形式，字符串模板与回调函数返回的字符串均支持用 \n 换行。
    # 模板变量有 {a}, {b}，{c}，{d}，{e}，分别表示系列名，数据名，数据值等。 
    # 在 trigger 为 'axis' 的时候，会有多个系列的数据，此时可以通过 {a0}, {a1}, {a2} 这种后面加索引的方式表示系列的索引。 
    # 不同图表类型下的 {a}，{b}，{c}，{d} 含义不一样。 其中变量{a}, {b}, {c}, {d}在不同图表类型下代表数据含义为：

    # 折线（区域）图、柱状（条形）图、K线图 : {a}（系列名称），{b}（类目值），{c}（数值）, {d}（无）
    # 散点图（气泡）图 : {a}（系列名称），{b}（数据名称），{c}（数值数组）, {d}（无）
    # 地图 : {a}（系列名称），{b}（区域名称），{c}（合并数值）, {d}（无）
    # 饼图、仪表盘、漏斗图: {a}（系列名称），{b}（数据项名称），{c}（数值）, {d}（百分比）
    # 示例：formatter: '{b}: {@score}'
    # 
    # 回调函数，回调函数格式：
    # (params: Object|Array) => string
    # 参数 params 是 formatter 需要的单个数据集。格式如下：
    # {
    #    componentType: 'series',
    #    // 系列类型
    #    seriesType: string,
    #    // 系列在传入的 option.series 中的 index
    #    seriesIndex: number,
    #    // 系列名称
    #    seriesName: string,
    #    // 数据名，类目名
    #    name: string,
    #    // 数据在传入的 data 数组中的 index
    #    dataIndex: number,
    #    // 传入的原始数据项
    #    data: Object,
    #    // 传入的数据值
    #    value: number|Array,
    #    // 数据图形的颜色
    #    color: string,
    # }
    formatter: Optional[str] = None,

    # 在 rich 里面，可以自定义富文本样式。利用富文本样式，可以在标签中做出非常丰富的效果
    # 具体配置可以参考一下 https://www.echartsjs.com/tutorial.html#%E5%AF%8C%E6%96%87%E6%9C%AC%E6%A0%87%E7%AD%BE
    rich: Optional[dict] = None,
)
```

## MarkPointItem - 标记点数据项

```python
class MarkPointItem(
    # 标注名称。
    name: Optional[str] = None,

    # 特殊的标注类型，用于标注最大值最小值等。可选:
    # 'min' 最大值。
    # 'max' 最大值。
    # 'average' 平均值。
    type_: Optional[str] = None,

    # 在使用 type 时有效，用于指定在哪个维度上指定最大值最小值，可以是 
    # 0（xAxis, radiusAxis），
    # 1（yAxis, angleAxis），默认使用第一个数值轴所在的维度。
    value_index: Optional[Numeric] = None,

    # 在使用 type 时有效，用于指定在哪个维度上指定最大值最小值。这可以是维度的直接名称，
    # 例如折线图时可以是 x、angle 等、candlestick 图时可以是 open、close 等维度名称。
    value_dim: Optional[str] = None,

    # 标注的坐标。坐标格式视系列的坐标系而定，可以是直角坐标系上的 x, y，
    # 也可以是极坐标系上的 radius, angle。例如 [121, 2323]、['aa', 998]。
    coord: Optional[Sequence] = None,

    # 相对容器的屏幕 x 坐标，单位像素。
    x: Optional[Numeric] = None,

    # 相对容器的屏幕 y 坐标，单位像素。
    y: Optional[Numeric] = None,

    # 标注值，可以不设。
    value: Optional[Numeric] = None,

    # 标记的图形。
    # ECharts 提供的标记类型包括 'circle', 'rect', 'roundRect', 'triangle', 
    # 'diamond', 'pin', 'arrow', 'none'
    # 可以通过 'image://url' 设置为图片，其中 URL 为图片的链接，或者 dataURI。
    symbol: Optional[str] = None,

    # 标记的大小，可以设置成诸如 10 这样单一的数字，也可以用数组分开表示宽和高，
    # 例如 [20, 10] 表示标记宽为 20，高为 10。
    symbol_size: Union[Numeric, Sequence] = None,

    # 标记点样式配置项，参考 `series_options.ItemStyleOpts`
    itemstyle_opts: Union[ItemStyleOpts, dict, None] = None,
)
```

## MarkPointOpts - 标记点

```python
class MarkPointOpts(
    # 标记点数据，参考 `series_options.MarkPointItem`
    data: Sequence[Union[MarkPointItem, dict]] = None,

    # 标记的图形。
    # ECharts 提供的标记类型包括 'circle', 'rect', 'roundRect', 'triangle', 
    # 'diamond', 'pin', 'arrow', 'none'
    # 可以通过 'image://url' 设置为图片，其中 URL 为图片的链接，或者 dataURI。
    symbol: Optional[str] = None,

    # 标记的大小，可以设置成诸如 10 这样单一的数字，也可以用数组分开表示宽和高，
    # 例如 [20, 10] 表示标记宽为 20，高为 10。
    # 如果需要每个数据的图形大小不一样，可以设置为如下格式的回调函数：
    # (value: Array|number, params: Object) => number|Array
    # 其中第一个参数 value 为 data 中的数据值。第二个参数 params 是其它的数据项参数。
    symbol_size: Union[None, Numeric] = None,

    # 标签配置项，参考 `series_options.LabelOpts`
    label_opts: LabelOpts = LabelOpts(position="inside", color="#fff"),
```

### e.g.

```python
bar = Bar()
bar.add_xaxis(list(x))
bar.add_yaxis('name', y)
bar.set_series_opts(
    label_opts = opts.LabelOpts(is_show = False),
    markpoint_opts = opts.MarkPointOpts(
        data = [
            opts.MarkPointItem(type = 'max', name = '最大值'),
            opts.MarkPointItem(type = 'min', name = '最小值'),
            opts.MarkPointItem(type = 'average', name = '平均值'),
            ]
        )
    )
```

![](./屏幕截图%202022-11-04%20000447.png)

## ItemStyleOpts - 图元样式

> 有一些系列配置项是放在其他地方的，这取决于此配置项用来修饰的对象；比如 `itemstyle_opts` 用来修饰 bar 的颜色时就放在 `add_yaxis` 里，要确定配置的使用位置
> 
> 比如 `add_yaxis` 这种函数包含哪些配置项建议先阅读每个图形最前面的 .class 说明

```python
class ItemStyleOpts(
    # 图形的颜色。
    # 颜色可以使用 RGB 表示，比如 'rgb(128, 128, 128)'，如果想要加上 alpha 通道表示不透明度，
    # 可以使用 RGBA，比如 'rgba(128, 128, 128, 0.5)'，也可以使用十六进制格式，比如 '#ccc'。
    # 除了纯色之外颜色也支持渐变色和纹理填充
    # 
    # 线性渐变，前四个参数分别是 x0, y0, x2, y2, 范围从 0 - 1，相当于在图形包围盒中的百分比，
    # 如果 globalCoord 为 `true`，则该四个值是绝对的像素位置
    # color: {
    #    type: 'linear',
    #    x: 0,
    #    y: 0,
    #    x2: 0,
    #    y2: 1,
    #    colorStops: [{
    #        offset: 0, color: 'red' // 0% 处的颜色
    #    }, {
    #        offset: 1, color: 'blue' // 100% 处的颜色
    #    }],
    #    global: false // 缺省为 false
    # }
    # 
    # 径向渐变，前三个参数分别是圆心 x, y 和半径，取值同线性渐变
    # color: {
    #    type: 'radial',
    #    x: 0.5,
    #    y: 0.5,
    #    r: 0.5,
    #    colorStops: [{
    #        offset: 0, color: 'red' // 0% 处的颜色
    #    }, {
    #        offset: 1, color: 'blue' // 100% 处的颜色
    #    }],
    #    global: false // 缺省为 false
    # }
    # 
    # 纹理填充
    # color: {
    #    image: imageDom, // 支持为 HTMLImageElement, HTMLCanvasElement，不支持路径字符串
    #    repeat: 'repeat' // 是否平铺, 可以是 'repeat-x', 'repeat-y', 'no-repeat'
    # }
    color: Optional[str] = None,

    # 阴线 图形的颜色。
    color0: Optional[str] = None,

    # 图形的描边颜色。支持的颜色格式同 color，不支持回调函数。
    border_color: Optional[str] = None,

    # 阴线 图形的描边颜色。
    border_color0: Optional[str] = None,

    # 描边宽度，默认不描边。
    border_width: Optional[Numeric] = None,

    # 支持 'dashed', 'dotted'。
    border_type: Optional[str] = None,

    # 图形透明度。支持从 0 到 1 的数字，为 0 时不绘制该图形。
    opacity: Optional[Numeric] = None,

    # 区域的颜色。    
    area_color: Optional[str] = None,
)
```

### e.g.

```python
# e.g.
bar = Bar()
bar.add_xaxis(list(x))
bar.add_yaxis('name', y, itemstyle_opts = opts.ItemStyleOpts(color = 'pink'))
```

![](./屏幕截图%202022-11-04%20001718.png)

# 主题定制

> 有些配置需要在图形函数中配置，比如主题的设定

```python
from pyecharts.globals import ThemeType

def theme_default() -> Bar:
    c = (
        Bar()
        # 等价于
        # Bar(init_opts=opts.InitOpts(theme=ThemeType.WHITE))
        .add_xaxis(Faker.choose())
        .add_yaxis("商家A", Faker.values())
        .add_yaxis("商家B", Faker.values())
        .add_yaxis("商家C", Faker.values())
        .add_yaxis("商家D", Faker.values())
        .set_global_opts(title_opts=opts.TitleOpts("Theme-default"))
    )
    return c
```

[定制主题 - pyecharts - A Python Echarts Plotting Library built with love.](https://pyecharts.org/#/zh-cn/themes)

# 数据类型问题

转换方式    `Series.tolist()`

# 饼图

```python
from pyecharts.charts import Page, Pie
fig = Pie().add('', [list(z) gor z in zip(p0.index.tolist(), p0.tolist())])
fig.set_series_opts(label_opts = opts.LabelOpts(is_show = False))
fig.set_global_opts(
    title_opts = opts.TitleOpts(title = 'Pie-Radius),
    legend_opts = opts.LegendOpts(
        orient = 'verical', pos_top = '15%', pos_left = '2%'
    )
)
```

# 多维数据

所谓多维，就是数据不仅仅有 x, y 两列，而是有多列数据特征需要展示；这里主要分为两类展示方法，一是用多张图展示多个数据，二是一张图展示多列数据

# 组合图表

[组合图表 - pyecharts - A Python Echarts Plotting Library built with love.](https://pyecharts.org/#/zh-cn/composite_charts)

## Grid - 并行多图

> Grid

```python
class Grid(
    # 初始化配置项，参考 `global_options.InitOpts`
    init_opts: opts.InitOpts = opts.InitOpts()
)
```

> add

```python
def add(
    # 图表实例，仅 `Chart` 类或者其子类
    chart: Chart,

    # 直角坐标系网格配置项，参见 `GridOpts`
    grid_opts: Union[opts.GridOpts, dict],

    # 直角坐标系网格索引
    grid_index: int = 0,

    # 是否由自己控制 Axis 索引
    is_control_axis_index: bool = False,
)
```

> GridOpts - 直角坐标系网格配置项

```python
class GridOpts(
    # 是否显示直角坐标系网格。
    is_show: bool = False,

    # 所有图形的 zlevel 值。
    z_level: Numeric = 0,

    # 组件的所有图形的z值。
    z: Numeric = 2,

    # grid 组件离容器左侧的距离。
    # left 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比，
    # 也可以是 'left', 'center', 'right'。
    # 如果 left 的值为'left', 'center', 'right'，组件会根据相应的位置自动对齐。
    pos_left: Union[Numeric, str, None] = None,

    # grid 组件离容器上侧的距离。
    # top 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比，
    # 也可以是 'top', 'middle', 'bottom'。
    # 如果 top 的值为'top', 'middle', 'bottom'，组件会根据相应的位置自动对齐。
    pos_top: Union[Numeric, str, None]  = None,

    # grid 组件离容器右侧的距离。
    # right 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比。
    pos_right: Union[Numeric, str, None]  = None,

    # grid 组件离容器下侧的距离。
    # bottom 的值可以是像 20 这样的具体像素值，可以是像 '20%' 这样相对于容器高宽的百分比。
    pos_bottom: Union[Numeric, str, None]  = None,

    # grid 组件的宽度。默认自适应。
    width: Union[Numeric, str, None] = None,

    # grid 组件的高度。默认自适应。
    height: Union[Numeric, str, None] = None,

    # grid 区域是否包含坐标轴的刻度标签。
    is_contain_label: bool = False,

    # 网格背景色，默认透明。
    background_color: str = "transparent",

    # 网格的边框颜色。支持的颜色格式同 backgroundColor。
    border_color: str = "#ccc",

    # 网格的边框线宽。
    border_width: Numeric = 1,

    # 本坐标系特定的 tooltip 设定。
    tooltip_opts: Union[TooltipOpts, dict, None] = None,
)
```

### e.g.

```python
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line
from pyecharts.faker import Faker

bar = (
    Bar()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(title_opts=opts.TitleOpts(title="Grid-Bar"))
)
line = (
    Line()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Grid-Line", pos_top="48%"),
        legend_opts=opts.LegendOpts(pos_top="48%"),
    )
)

grid = (
    Grid()
    .add(bar, grid_opts=opts.GridOpts(pos_bottom="60%"))
    .add(line, grid_opts=opts.GridOpts(pos_top="60%"))
    .render("grid_vertical.html")
)
```

![](./屏幕截图%202022-11-04%20011755.png)

---

```python
from pyecharts import options as opts
from pyecharts.charts import Bar, Geo, Grid
from pyecharts.faker import Faker

bar = (
    Bar()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(legend_opts=opts.LegendOpts(pos_left="20%"))
)
geo = (
    Geo()
    .add_schema(maptype="china")
    .add("geo", [list(z) for z in zip(Faker.provinces, Faker.values())])
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        visualmap_opts=opts.VisualMapOpts(),
        title_opts=opts.TitleOpts(title="Grid-Geo-Bar"),
    )
)

grid = (
    Grid()
    .add(bar, grid_opts=opts.GridOpts(pos_top="50%", pos_right="75%"))
    .add(geo, grid_opts=opts.GridOpts(pos_left="60%"))
    .render("grid_geo_bar.html")
)
```

![](./屏幕截图%202022-11-04%20011849.png)

---

```python
from pyecharts import options as opts
from pyecharts.charts import Grid, Line, Scatter
from pyecharts.faker import Faker

scatter = (
    Scatter()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Grid-Scatter"),
        legend_opts=opts.LegendOpts(pos_left="20%"),
    )
)
line = (
    Line()
    .add_xaxis(Faker.choose())
    .add_yaxis("商家A", Faker.values())
    .add_yaxis("商家B", Faker.values())
    .set_global_opts(
        title_opts=opts.TitleOpts(title="Grid-Line", pos_right="5%"),
        legend_opts=opts.LegendOpts(pos_right="20%"),
    )
)

grid = (
    Grid()
    .add(scatter, grid_opts=opts.GridOpts(pos_left="55%"))
    .add(line, grid_opts=opts.GridOpts(pos_right="55%"))
    .render("grid_horizontal.html")
)
```

![](./屏幕截图%202022-11-04%20011849.png)

## Page - 顺序多图

> Page

```python
class Page(
    # HTML 标题
    page_title: str = "Awesome-pyecharts",

    # 远程 HOST，默认为 "https://assets.pyecharts.org/assets/"
    js_host: str = "",

    # 每个图例之间的间隔
    interval: int = 1,

    # 布局配置项，参考 `PageLayoutOpts`
    layout: Union[PageLayoutOpts, dict] = PageLayoutOpts()
)
```

> add

```python
def add(*charts)    # charts: 任意图表实例
```

> PageLayoutOpts

```python
class PageLayoutOpts(
    # 配置均为原生 CSS 样式
    justify_content: Optional[str] = None,
    margin: Optional[str] = None,
    display: Optional[str] = None,
    flex_wrap: Optional[str] = None,
)
```

> save_resize_html
> 
> 用于 DraggablePageLayout

```python
def save_resize_html(
    # Page 第一次渲染后的 html 文件
    source: str = "render.html",
    *,
    # 布局配置文件
    cfg_file: types.Optional[str] = None,

    # 布局配置 dict
    cfg_dict: types.Optional[list] = None,

    # 重新生成的 .html 存放路径
    dest: str = "resize_render.html",
) -> str
```

Page 内置了以下布局

- SimplePageLayout

- DraggablePageLayout

<mark>默认布局</mark>

```python
page = Page()
page.add(bar_datazoom_slider(), line_markpoint(), pie_rosetype(), grid_mutil_yaxis())
page.render()
```

<mark>SimplePageLayout 布局</mark>

```python
page = Page(layout=Page.SimplePageLayout)
# 需要自行调整每个 chart 的 height/width，显示效果在不同的显示器上可能不同
page.add(bar_datazoom_slider(), line_markpoint(), pie_rosetype(), grid_mutil_yaxis())
page.render()
```

![](./屏幕截图%202022-11-04%20012749.png)

<mark>DraggablePageLayout 布局</mark>

```python
page = Page(layout=Page.DraggablePageLayout)
page.add(bar_datazoom_slider(), line_markpoint(), pie_rosetype(), grid_mutil_yaxis())
page.render()
```

## Tab - 选项卡多图

> Tab

```python
class Tab(
    # HTML 标题
    page_title: str = "Awesome-pyecharts",

    # 远程 HOST，默认为 "https://assets.pyecharts.org/assets/"
    js_host: str = ""
)
```

> add

```python
def add(
    # 任意图表类型
    chart,

    # 标签名称
    tab_name
):
```

### e.g.

```python
from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Line, Pie, Tab
from pyecharts.faker import Faker


def bar_datazoom_slider() -> Bar:
    c = (
        Bar()
        .add_xaxis(Faker.days_attrs)
        .add_yaxis("商家A", Faker.days_values)
        .set_global_opts(
            title_opts=opts.TitleOpts(title="Bar-DataZoom（slider-水平）"),
            datazoom_opts=[opts.DataZoomOpts()],
        )
    )
    return c


def line_markpoint() -> Line:
    c = (
        Line()
        .add_xaxis(Faker.choose())
        .add_yaxis(
            "商家A",
            Faker.values(),
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="min")]),
        )
        .add_yaxis(
            "商家B",
            Faker.values(),
            markpoint_opts=opts.MarkPointOpts(data=[opts.MarkPointItem(type_="max")]),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="Line-MarkPoint"))
    )
    return c


def pie_rosetype() -> Pie:
    v = Faker.choose()
    c = (
        Pie()
        .add(
            "",
            [list(z) for z in zip(v, Faker.values())],
            radius=["30%", "75%"],
            center=["25%", "50%"],
            rosetype="radius",
            label_opts=opts.LabelOpts(is_show=False),
        )
        .add(
            "",
            [list(z) for z in zip(v, Faker.values())],
            radius=["30%", "75%"],
            center=["75%", "50%"],
            rosetype="area",
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="Pie-玫瑰图示例"))
    )
    return c


def grid_mutil_yaxis() -> Grid:
    x_data = ["{}月".format(i) for i in range(1, 13)]
    bar = (
        Bar()
        .add_xaxis(x_data)
        .add_yaxis(
            "蒸发量",
            [2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3],
            yaxis_index=0,
            color="#d14a61",
        )
        .add_yaxis(
            "降水量",
            [2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3],
            yaxis_index=1,
            color="#5793f3",
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                name="蒸发量",
                type_="value",
                min_=0,
                max_=250,
                position="right",
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#d14a61")
                ),
                axislabel_opts=opts.LabelOpts(formatter="{value} ml"),
            )
        )
        .extend_axis(
            yaxis=opts.AxisOpts(
                type_="value",
                name="温度",
                min_=0,
                max_=25,
                position="left",
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#675bba")
                ),
                axislabel_opts=opts.LabelOpts(formatter="{value} °C"),
                splitline_opts=opts.SplitLineOpts(
                    is_show=True, linestyle_opts=opts.LineStyleOpts(opacity=1)
                ),
            )
        )
        .set_global_opts(
            yaxis_opts=opts.AxisOpts(
                name="降水量",
                min_=0,
                max_=250,
                position="right",
                offset=80,
                axisline_opts=opts.AxisLineOpts(
                    linestyle_opts=opts.LineStyleOpts(color="#5793f3")
                ),
                axislabel_opts=opts.LabelOpts(formatter="{value} ml"),
            ),
            title_opts=opts.TitleOpts(title="Grid-多 Y 轴示例"),
            tooltip_opts=opts.TooltipOpts(trigger="axis", axis_pointer_type="cross"),
        )
    )

    line = (
        Line()
        .add_xaxis(x_data)
        .add_yaxis(
            "平均温度",
            [2.0, 2.2, 3.3, 4.5, 6.3, 10.2, 20.3, 23.4, 23.0, 16.5, 12.0, 6.2],
            yaxis_index=2,
            color="#675bba",
            label_opts=opts.LabelOpts(is_show=False),
        )
    )

    bar.overlap(line)
    return Grid().add(
        bar, opts.GridOpts(pos_left="5%", pos_right="20%"), is_control_axis_index=True
    )


tab = Tab()
tab.add(bar_datazoom_slider(), "bar-example")
tab.add(line_markpoint(), "line-example")
tab.add(pie_rosetype(), "pie-example")
tab.add(grid_mutil_yaxis(), "grid-example")
tab.render("tab_base.html")
```

![](./屏幕截图%202022-11-04%20013220.png)

# 数字大屏

```python
'''
制作标题
'''
title = Pie().set_global_opts(title_opts=opts.TitleOpts(title="2021疫情数据大屏", title_textstyle_opts=opts.TextStyleOpts(font_size=40, color='#FFFF99'), pos_top=0))
title.render_notebook()

subtitle = Pie().set_global_opts(title_opts=opts.TitleOpts(subtitle=(subtime),
                                                           subtitle_textstyle_opts=opts.TextStyleOpts(font_size=15, color='#FFFF99'),
                                                           pos_top=0
                                                          )
                                )
subtitle.render_notebook()
```

```python
page = Page(layout=Page.DraggablePageLayout, page_title='2021疫情数据大屏')
page.add(  # 你需要拼接大屏的图
         table,
         pie1,
         map1,
         map2,
         heatmap,
         map3, 
         table1,
         bar1,
         line1,
         wc,
         title,
         subtitle

        )
page.render()
```

**运行后得到 render.html 打开后会看到左上角有一个控件，并且每个小图都是可移动的。按照你喜欢的排版设置好后，点击控件即可保存得到一个 .json 的文件**

```python
Page.save_resize_html("render.html", cfg_file=r"chart_config.json", dest="my_new_charts.html");
```

**得到 .json 文件后再运行以上代码即可得到数据大屏了**
