# 失败案例库（自动生成）

样本总数 **212**。模拟一个「训练到中途」的 agent：主流程已学会，但各类工程/能力缺陷仍在。

| 失败类型 | 含义 | 数量 | 占比 |
|---|---|---|---|
| `correct` | 答对 | 87 | 41.0% |
| `format_error` | 输出无合法 <search>/<answer> 标签，格式未学会 | 21 | 9.9% |
| `no_search` | 未调用检索即作答，依赖参数记忆 | 31 | 14.6% |
| `over_search` | 检索预算（B=4）耗尽仍未收敛到答案 | 14 | 6.6% |
| `retrieval_miss` | 检索了但 top-k 未召回支撑文档，检索器/索引问题 | 17 | 8.0% |
| `query_poor` | 检索 query 与问题几乎无重叠，query 生成能力不足 | 0 | 0.0% |
| `unfaithful` | 检索命中支撑文档但答案无法由检索内容推出，幻觉 | 17 | 8.0% |
| `aggregation_error` | 检索命中且答案相关，但聚合/数值/多跳推理错误 | 24 | 11.3% |
| `gold_strict` | 语义正确但严格 EM 判错，属评测口径问题 | 1 | 0.5% |

## `format_error` — 输出无合法 <search>/<answer> 标签，格式未学会

共 21 例，下面取 3 例。

**Q**: 《为什么你炒的糖色总是发苦》的点赞数是多少？

- gold: `['4.10万', '4.1万', '41000']`
- pred: ``
- 检索次数 0｜命中文档 []｜支撑文档 ['bili_012']
- 归因理由: 输出中没有可解析的 <search> 或 <answer> 标签
- 触发行为: `format_fail`

<details><summary>完整轨迹</summary>

```
<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.
```

</details>

**Q**: 哔哩哔哩「健身」话题下点赞数最高的内容，UP主是谁？

- gold: `['铁馆阿泽']`
- pred: ``
- 检索次数 0｜命中文档 []｜支撑文档 ['bili_007', 'bili_008']
- 归因理由: 输出中没有可解析的 <search> 或 <answer> 标签
- 触发行为: `format_fail`

<details><summary>完整轨迹</summary>

```
<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.
```

</details>

**Q**: 《把机械键盘拆开之后，我理解了什么叫手感》发布于哪一天？

- gold: `['2025-02-18', '2025/02/18', '2025年2月18日']`
- pred: ``
- 检索次数 0｜命中文档 []｜支撑文档 ['bili_003']
- 归因理由: 输出中没有可解析的 <search> 或 <answer> 标签
- 触发行为: `format_fail`

<details><summary>完整轨迹</summary>

```
<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.<think> I think I know this already. </think> The answer should be obvious. My action is not correct. Let me rethink.
```

</details>

## `no_search` — 未调用检索即作答，依赖参数记忆

共 31 例，下面取 3 例。

**Q**: 「AI工具」话题下，哪个平台这些内容的点赞总数最高？

- gold: `['抖音', 'douyin']`
- pred: `哔哩哔哩`
- 检索次数 0｜命中文档 []｜支撑文档 ['bili_004', 'bili_005', 'bili_006', 'dy_010', 'dy_011', 'xhs_014']
- 归因理由: 未调用检索即给出答案
- 触发行为: `no_search`

<details><summary>完整轨迹</summary>

```
<think> I recall this from memory. </think><answer> 哔哩哔哩 </answer>
```

</details>

**Q**: 小红书「健身」话题下点赞数最高的内容，博主是谁？

- gold: `['在家练的老张']`
- pred: `慢旅行阿元`
- 检索次数 0｜命中文档 []｜支撑文档 ['xhs_012', 'xhs_013']
- 归因理由: 未调用检索即给出答案
- 触发行为: `no_search`

<details><summary>完整轨迹</summary>

```
<think> I recall this from memory. </think><answer> 慢旅行阿元 </answer>
```

</details>

**Q**: 小红书上博主「虚空博主乙」发布的内容里，曝光量最高的是哪一条？请回答完整标题。

- gold: `['信息不足']`
- pred: `通勤穿搭十套不重样，预算控制在两千内`
- 检索次数 0｜命中文档 []｜支撑文档 []
- 归因理由: 未调用检索即给出答案
- 触发行为: `no_search`

<details><summary>完整轨迹</summary>

```
<think> I recall this from memory. </think><answer> 通勤穿搭十套不重样，预算控制在两千内 </answer>
```

</details>

## `over_search` — 检索预算（B=4）耗尽仍未收敛到答案

共 14 例，下面取 3 例。

**Q**: 《深蹲膝盖疼？先看完这四种常见错误》发布于哪一天？

- gold: `['2025-03-30', '2025/03/30', '2025年3月30日']`
- pred: ``
- 检索次数 4｜命中文档 ['bili_008', 'dy_004', 'xhs_014']｜支撑文档 ['bili_008']
- 归因理由: 检索 4 次后仍未产出 <answer>
- 触发行为: `over_search`

<details><summary>完整轨迹</summary>

```
<think> I still need more information, let me search again. </think><search> 《深蹲膝盖疼？先看完这四种常见错误》发布于哪一天？ (attempt 1) </search><information>Doc 1(Title: "深蹲膝盖疼？先看完这四种常见错误") 平台：哔哩哔哩；话题：健身；UP主：康复师小林；发布日期：2025-03-30；播放量：890000；点赞数：39000；投币数：52000；收藏数：9700；评论数：9700；时长：767秒；简介：从解剖学角度解释膝内扣与重心前移 Doc 2(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板 Doc 3(Title: "一天走完一座小城，路线我排好了") 平台：抖音；话题：旅行；作者：周末去哪儿；发布日期：2025-03-22；播放量：6210000；点赞数：376000；分享数：69000；评论数：34000；时长：55秒；简介：时间分配与交通衔接</information><think> I still need more information, let me search again. </think><search> 《深蹲膝盖疼？先看完这四种常见错误》发布于哪一天？ (attempt 2) </search><information>Doc 1(Title: "深蹲膝盖疼？先看完这四种常见错误") 平台：哔哩哔哩；话题：健身；UP主：康复师小林；发布日期：2025-03-30；播放量：890000；点赞数：39000；投币数：52000；收藏数：9700；评论数：9700；时长：767秒；简介：从解剖学角度解释膝内扣与重心前移 Doc 2(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板 Doc 3(Title: "一天走完一座小城，路线我排好了") 平台：抖音；话题：旅行；作者：周末去哪儿；发布日期：2025-03-22；播放量：6210000；点赞数：376000；分享数：69000；评论数：34000；时长：55秒；简介：时间分配与交通衔接</information><think> I still need more information, let me search again. </think><search> 《深蹲膝盖疼？先看完这四种常见错误》发布于哪一天？ (attempt 3) </search><information>Doc 1(Title: "深蹲膝盖疼？先看完这四种常见错误") 平台：哔哩哔哩；话题：健身；UP主：康复师小林；发布日期：2025-03-30；播放量：890000；点赞数：39000；投币数：52000；收藏数：9700；评论数：9700；时长：767秒；简介：从解剖学角度解释膝内扣与重心前移 Doc 2(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板 Doc 3(Title: "一天走完一座小城，路线我排好了") 平台：抖音；话题：旅行；作者：周末去哪儿；发布日期：2025-03-22；播放量：6210000；点赞数：376000；分享数：69000；评论数：34000；时长：55秒；简介：时间分配与交通衔接</information><think> I still need more information, let me search again. </think><search> 《深蹲膝盖疼？先看完这四种
```

</details>

**Q**: 抖音「健身」话题下一共有多少条内容？

- gold: `['2']`
- pred: ``
- 检索次数 4｜命中文档 ['dy_006', 'dy_007', 'xhs_014']｜支撑文档 ['dy_006', 'dy_007']
- 归因理由: 检索 4 次后仍未产出 <answer>
- 触发行为: `over_search`

<details><summary>完整轨迹</summary>

```
<think> I still need more information, let me search again. </think><search> 抖音「健身」话题下一共有多少条内容？ (attempt 1) </search><information>Doc 1(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板 Doc 2(Title: "每天十分钟，把久坐的腰救回来") 平台：抖音；话题：健身；作者：办公桌健身；发布日期：2025-01-30；播放量：5470000；点赞数：331000；分享数：74000；评论数：38000；时长：46秒；简介：三个动作的发力要点 Doc 3(Title: "健身房里90%的人都做错了这个动作") 平台：抖音；话题：健身；作者：硬核教练阿凯；发布日期：2025-04-29；播放量：6820000；点赞数：402000；分享数：93000；评论数：61000；时长：51秒；简介：常见代偿与纠正方式</information><think> I still need more information, let me search again. </think><search> 抖音「健身」话题下一共有多少条内容？ (attempt 2) </search><information>Doc 1(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板 Doc 2(Title: "每天十分钟，把久坐的腰救回来") 平台：抖音；话题：健身；作者：办公桌健身；发布日期：2025-01-30；播放量：5470000；点赞数：331000；分享数：74000；评论数：38000；时长：46秒；简介：三个动作的发力要点 Doc 3(Title: "健身房里90%的人都做错了这个动作") 平台：抖音；话题：健身；作者：硬核教练阿凯；发布日期：2025-04-29；播放量：6820000；点赞数：402000；分享数：93000；评论数：61000；时长：51秒；简介：常见代偿与纠正方式</information><think> I still need more information, let me search again. </think><search> 抖音「健身」话题下一共有多少条内容？ (attempt 3) </search><information>Doc 1(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板 Doc 2(Title: "每天十分钟，把久坐的腰救回来") 平台：抖音；话题：健身；作者：办公桌健身；发布日期：2025-01-30；播放量：5470000；点赞数：331000；分享数：74000；评论数：38000；时长：46秒；简介：三个动作的发力要点 Doc 3(Title: "健身房里90%的人都做错了这个动作") 平台：抖音；话题：健身；作者：硬核教练阿凯；发布日期：2025-04-29；播放量：6820000；点赞数：402000；分享数：93000；评论数：61000；时长：51秒；简介：常见代偿与纠正方式</information><think> I still need more information, let me search again. </think><search> 抖音「健身」话题下一共有多少条内容？ (attempt 4) </search><information>Doc 1(Title: "用 AI 做小红书封面，点击率
```

</details>

**Q**: 哔哩哔哩「健身」话题下这些内容的平均点赞数是多少？

- gold: `['8.00万', '8.0万', '80000']`
- pred: ``
- 检索次数 4｜命中文档 ['bili_007', 'bili_008', 'xhs_014']｜支撑文档 ['bili_007', 'bili_008']
- 归因理由: 检索 4 次后仍未产出 <answer>
- 触发行为: `over_search`

<details><summary>完整轨迹</summary>

```
<think> I still need more information, let me search again. </think><search> 哔哩哔哩「健身」话题下这些内容的平均点赞数是多少？ (attempt 1) </search><information>Doc 1(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板 Doc 2(Title: "三个月徒手训练，我的体脂率从26%到17%") 平台：哔哩哔哩；话题：健身；UP主：铁馆阿泽；发布日期：2025-01-25；播放量：2680000；点赞数：121000；投币数：78000；收藏数：34000；评论数：34000；时长：932秒；简介：训练计划与饮食控制的完整记录 Doc 3(Title: "深蹲膝盖疼？先看完这四种常见错误") 平台：哔哩哔哩；话题：健身；UP主：康复师小林；发布日期：2025-03-30；播放量：890000；点赞数：39000；投币数：52000；收藏数：9700；评论数：9700；时长：767秒；简介：从解剖学角度解释膝内扣与重心前移</information><think> I still need more information, let me search again. </think><search> 哔哩哔哩「健身」话题下这些内容的平均点赞数是多少？ (attempt 2) </search><information>Doc 1(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板 Doc 2(Title: "三个月徒手训练，我的体脂率从26%到17%") 平台：哔哩哔哩；话题：健身；UP主：铁馆阿泽；发布日期：2025-01-25；播放量：2680000；点赞数：121000；投币数：78000；收藏数：34000；评论数：34000；时长：932秒；简介：训练计划与饮食控制的完整记录 Doc 3(Title: "深蹲膝盖疼？先看完这四种常见错误") 平台：哔哩哔哩；话题：健身；UP主：康复师小林；发布日期：2025-03-30；播放量：890000；点赞数：39000；投币数：52000；收藏数：9700；评论数：9700；时长：767秒；简介：从解剖学角度解释膝内扣与重心前移</information><think> I still need more information, let me search again. </think><search> 哔哩哔哩「健身」话题下这些内容的平均点赞数是多少？ (attempt 3) </search><information>Doc 1(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板 Doc 2(Title: "三个月徒手训练，我的体脂率从26%到17%") 平台：哔哩哔哩；话题：健身；UP主：铁馆阿泽；发布日期：2025-01-25；播放量：2680000；点赞数：121000；投币数：78000；收藏数：34000；评论数：34000；时长：932秒；简介：训练计划与饮食控制的完整记录 Doc 3(Title: "深蹲膝盖疼？先看完这四种常见错误") 平台：哔哩哔哩；话题：健身；UP主：康复师小林；发布日期：2025-03-30；播放量：890000；点赞数：39000；投币数：52000；收藏数：9700；评论数：9700；时长：767秒；简介：从解剖学角度解释膝内扣与重心前移</information><think> I still ne
```

</details>

## `retrieval_miss` — 检索了但 top-k 未召回支撑文档，检索器/索引问题

共 17 例，下面取 3 例。

**Q**: 抖音「美食」话题下这些内容的平均点赞数是多少？

- gold: `['36.57万', '36.6万', '365667']`
- pred: `301000`
- 检索次数 1｜命中文档 ['bili_001', 'bili_002', 'bili_003']｜支撑文档 ['dy_001', 'dy_002', 'dy_003']
- 归因理由: top-k 结果中没有任何支撑文档
- 触发行为: `bad_query`

<details><summary>完整轨迹</summary>

```
<think> Let me search for something. </think><search> 今天天气怎么样 </search><information>Doc 1(Title: "一千五价位显示器怎么选？六台实测色域覆盖") 平台：哔哩哔哩；话题：数码测评；UP主：硬件茶谈；发布日期：2025-04-02；播放量：1240000；点赞数：52000；投币数：48000；收藏数：11000；评论数：11000；时长：892秒；简介：色域覆盖率实测数据与出厂校色报告 Doc 2(Title: "上岸之后我才明白的三件事") 平台：抖音；话题：考研；作者：考研过来人；发布日期：2025-05-13；播放量：3480000；点赞数：213000；分享数：71000；评论数：36000；时长：57秒；简介：信息差、心态与选择 Doc 3(Title: "这个隐藏关，百分之九十九的人没打过") 平台：抖音；话题：游戏；作者：游戏考古队；发布日期：2025-02-05；播放量：6670000；点赞数：389000；分享数：102000；评论数：52000；时长：88秒；简介：触发条件与路线</information><think> I have enough now. </think><answer> 301000 </answer>
```

</details>

**Q**: 小红书上标题为《用 AI 做小红书封面，点击率翻倍的 prompt》的内容，博主是谁？

- gold: `['AI副业日记']`
- pred: `硬件茶谈`
- 检索次数 1｜命中文档 ['bili_001', 'bili_002', 'bili_003']｜支撑文档 ['xhs_014']
- 归因理由: top-k 结果中没有任何支撑文档
- 触发行为: `bad_query`

<details><summary>完整轨迹</summary>

```
<think> Let me search for something. </think><search> 今天天气怎么样 </search><information>Doc 1(Title: "一千五价位显示器怎么选？六台实测色域覆盖") 平台：哔哩哔哩；话题：数码测评；UP主：硬件茶谈；发布日期：2025-04-02；播放量：1240000；点赞数：52000；投币数：48000；收藏数：11000；评论数：11000；时长：892秒；简介：色域覆盖率实测数据与出厂校色报告 Doc 2(Title: "上岸之后我才明白的三件事") 平台：抖音；话题：考研；作者：考研过来人；发布日期：2025-05-13；播放量：3480000；点赞数：213000；分享数：71000；评论数：36000；时长：57秒；简介：信息差、心态与选择 Doc 3(Title: "这个隐藏关，百分之九十九的人没打过") 平台：抖音；话题：游戏；作者：游戏考古队；发布日期：2025-02-05；播放量：6670000；点赞数：389000；分享数：102000；评论数：52000；时长：88秒；简介：触发条件与路线</information><think> I have enough now. </think><answer> 硬件茶谈 </answer>
```

</details>

**Q**: 抖音上「美妆」话题里播放量最高的内容标题是什么？

- gold: `['三十秒学会一个通勤妆']`
- pred: `这个隐藏关，百分之九十九的人没打过`
- 检索次数 1｜命中文档 ['bili_001', 'bili_002', 'bili_003']｜支撑文档 ['dy_012', 'dy_013']
- 归因理由: top-k 结果中没有任何支撑文档
- 触发行为: `bad_query`

<details><summary>完整轨迹</summary>

```
<think> Let me search for something. </think><search> 今天天气怎么样 </search><information>Doc 1(Title: "一千五价位显示器怎么选？六台实测色域覆盖") 平台：哔哩哔哩；话题：数码测评；UP主：硬件茶谈；发布日期：2025-04-02；播放量：1240000；点赞数：52000；投币数：48000；收藏数：11000；评论数：11000；时长：892秒；简介：色域覆盖率实测数据与出厂校色报告 Doc 2(Title: "上岸之后我才明白的三件事") 平台：抖音；话题：考研；作者：考研过来人；发布日期：2025-05-13；播放量：3480000；点赞数：213000；分享数：71000；评论数：36000；时长：57秒；简介：信息差、心态与选择 Doc 3(Title: "这个隐藏关，百分之九十九的人没打过") 平台：抖音；话题：游戏；作者：游戏考古队；发布日期：2025-02-05；播放量：6670000；点赞数：389000；分享数：102000；评论数：52000；时长：88秒；简介：触发条件与路线</information><think> I have enough now. </think><answer> 这个隐藏关，百分之九十九的人没打过 </answer>
```

</details>

## `unfaithful` — 检索命中支撑文档但答案无法由检索内容推出，幻觉

共 17 例，下面取 3 例。

**Q**: 小红书「穿搭」话题下点赞数最高的内容，博主是谁？

- gold: `['小个子研究所']`
- pred: `试色狂魔阿宁`
- 检索次数 1｜命中文档 ['xhs_004', 'xhs_005', 'xhs_014']｜支撑文档 ['xhs_004', 'xhs_005']
- 归因理由: 支撑文档已召回，但答案实词无法在检索文本中找到
- 触发行为: `unfaithful`

<details><summary>完整轨迹</summary>

```
<think> Let me look this up. </think><search> 小红书「穿搭」话题下点赞数最高的内容，博主是谁？ </search><information>Doc 1(Title: "通勤穿搭十套不重样，预算控制在两千内") 平台：小红书；话题：穿搭；博主：打工人衣柜；发布日期：2025-05-16；曝光量：417000；点赞数：34500；收藏数：40800；评论数：3300；简介：基础款为主、配饰做变化 Doc 2(Title: "155小个子显高穿搭，比例比身高重要") 平台：小红书；话题：穿搭；博主：小个子研究所；发布日期：2025-02-22；曝光量：894000；点赞数：75300；收藏数：61200；评论数：6800；简介：腰线位置与裤长、鞋型的三个变量 Doc 3(Title: "用 AI 做小红书封面，点击率翻倍的 prompt") 平台：小红书；话题：AI工具；博主：AI副业日记；发布日期：2025-06-30；曝光量：923000；点赞数：81700；收藏数：96300；评论数：7400；简介：给出可复制的提示词模板</information><think> I remember the answer. </think><answer> 试色狂魔阿宁 </answer>
```

</details>

**Q**: 哔哩哔哩「数码测评」话题下点赞数最高的内容，UP主是谁？

- gold: `['极客湾Geekerwan']`
- pred: `考研过来人`
- 检索次数 1｜命中文档 ['bili_001', 'bili_002', 'bili_003']｜支撑文档 ['bili_001', 'bili_002', 'bili_003']
- 归因理由: 支撑文档已召回，但答案实词无法在检索文本中找到
- 触发行为: `unfaithful`

<details><summary>完整轨迹</summary>

```
<think> Let me look this up. </think><search> 哔哩哔哩「数码测评」话题下点赞数最高的内容，UP主是谁？ </search><information>Doc 1(Title: "把机械键盘拆开之后，我理解了什么叫手感") 平台：哔哩哔哩；话题：数码测评；UP主：键盘侠老王；发布日期：2025-02-18；播放量：760000；点赞数：31000；投币数：22000；收藏数：8600；评论数：8600；时长：654秒；简介：拆解轴体结构与键帽材质对声音的影响 Doc 2(Title: "一千五价位显示器怎么选？六台实测色域覆盖") 平台：哔哩哔哩；话题：数码测评；UP主：硬件茶谈；发布日期：2025-04-02；播放量：1240000；点赞数：52000；投币数：48000；收藏数：11000；评论数：11000；时长：892秒；简介：色域覆盖率实测数据与出厂校色报告 Doc 3(Title: "2025年旗舰机横评：五台真机三个月的续航真相") 平台：哔哩哔哩；话题：数码测评；UP主：极客湾Geekerwan；发布日期：2025-03-12；播放量：3820000；点赞数：156000；投币数：91000；收藏数：42000；评论数：42000；时长：1180秒；简介：重点对比了续航与温控，结论是能效比优先于峰值性能</information><think> I remember the answer. </think><answer> 考研过来人 </answer>
```

</details>

**Q**: 《厨房台面收纳的六个尺寸，装之前必须量》和《四十平出租屋改造，总共花了六千八》都在小红书发布，哪一条的曝光量更高？请回答完整标题。

- gold: `['四十平出租屋改造，总共花了六千八']`
- pred: `海拔4700米，我拍到了最干净的一次星空`
- 检索次数 1｜命中文档 ['bili_009', 'xhs_006', 'xhs_007']｜支撑文档 ['xhs_006', 'xhs_007']
- 归因理由: 支撑文档已召回，但答案实词无法在检索文本中找到
- 触发行为: `unfaithful`

<details><summary>完整轨迹</summary>

```
<think> Let me look this up. </think><search> 《厨房台面收纳的六个尺寸，装之前必须量》和《四十平出租屋改造，总共花了六千八》都在小红书发布，哪一条的曝光量更高？请回答完整标题。 </search><information>Doc 1(Title: "厨房台面收纳的六个尺寸，装之前必须量") 平台：小红书；话题：家居；博主：住小帮阿May；发布日期：2025-04-25；曝光量：356000；点赞数：29800；收藏数：37600；评论数：2100；简介：给出台面深度与吊柜高度的实测值 Doc 2(Title: "四十平出租屋改造，总共花了六千八") 平台：小红书；话题：家居；博主：租房也要好好住；发布日期：2025-03-28；曝光量：1260000；点赞数：108000；收藏数：132000；评论数：9700；简介：收纳、灯光、软装的花费明细 Doc 3(Title: "一个人在关西走了十天，总共花了四千三") 平台：哔哩哔哩；话题：旅行；UP主：在路上阿May；发布日期：2025-04-19；播放量：1120000；点赞数：56000；投币数：44000；收藏数：15000；评论数：15000；时长：1355秒；简介：交通、住宿、门票的详细账单拆解</information><think> I remember the answer. </think><answer> 海拔4700米，我拍到了最干净的一次星空 </answer>
```

</details>

## `aggregation_error` — 检索命中且答案相关，但聚合/数值/多跳推理错误

共 24 例，下面取 3 例。

**Q**: 抖音上内容的「观众完播率」字段，平均值是多少？

- gold: `['信息不足']`
- pred: `抖音`
- 检索次数 2｜命中文档 ['bili_001', 'bili_002', 'bili_003', 'dy_001', 'dy_012', 'dy_015']｜支撑文档 []
- 归因理由: 检索命中，但聚合/数值/多跳推理出错
- 触发行为: `ideal`

<details><summary>完整轨迹</summary>

```
<think> I need external information for this. </think><search> 抖音上内容的「观众完播率」字段，平均值是多少？ </search><information>Doc 1(Title: "三十秒学会一个通勤妆") 平台：抖音；话题：美妆；作者：通勤妆阿May；发布日期：2025-03-09；播放量：5230000；点赞数：318000；分享数：58000；评论数：31000；时长：34秒；简介：分区上妆顺序 Doc 2(Title: "一口锅搞定一周的减脂餐") 平台：抖音；话题：美食；作者：减脂厨房；发布日期：2025-02-27；播放量：5820000；点赞数：342000；分享数：62000；评论数：41000；时长：58秒；简介：备餐分装与复热方法 Doc 3(Title: "上岸之后我才明白的三件事") 平台：抖音；话题：考研；作者：考研过来人；发布日期：2025-05-13；播放量：3480000；点赞数：213000；分享数：71000；评论数：36000；时长：57秒；简介：信息差、心态与选择</information><think> I still lack part of the information. </think><search> unknown 内容列表 </search><information>Doc 1(Title: "上岸之后我才明白的三件事") 平台：抖音；话题：考研；作者：考研过来人；发布日期：2025-05-13；播放量：3480000；点赞数：213000；分享数：71000；评论数：36000；时长：57秒；简介：信息差、心态与选择 Doc 2(Title: "这个隐藏关，百分之九十九的人没打过") 平台：抖音；话题：游戏；作者：游戏考古队；发布日期：2025-02-05；播放量：6670000；点赞数：389000；分享数：102000；评论数：52000；时长：88秒；简介：触发条件与路线 Doc 3(Title: "成分党看配方表，第一个要看的就是它") 平台：抖音；话题：美妆；作者：配方拆解阿宁；发布日期：2025-04-17；播放量：2910000；点赞数：176000；分享数：33000；评论数：19000；时长：73秒；简介：活性成分排序与浓度判断</information><think> Now I can answer based on the retrieved evidence. </think><answer> 抖音 </answer>
```

</details>

**Q**: 哔哩哔哩上标题为《在家复刻兰州牛肉面，汤头熬了六十个钟头》的内容，UP主是谁？

- gold: `['信息不足']`
- pred: `厨房实验室`
- 检索次数 2｜命中文档 ['bili_001', 'bili_002', 'bili_003', 'bili_011', 'xhs_006', 'xhs_012']｜支撑文档 []
- 归因理由: 检索命中，但聚合/数值/多跳推理出错
- 触发行为: `ideal`

<details><summary>完整轨迹</summary>

```
<think> I need external information for this. </think><search> 哔哩哔哩上标题为《在家复刻兰州牛肉面，汤头熬了六十个钟头》的内容，UP主是谁？ </search><information>Doc 1(Title: "在家复刻兰州牛肉面，汤头熬了六个钟头") 平台：哔哩哔哩；话题：美食；UP主：厨房实验室；发布日期：2025-05-21；播放量：1780000；点赞数：82000；投币数：97000；收藏数：19000；评论数：19000；时长：1104秒；简介：牛骨与香料配比、蓬灰水替代方案 Doc 2(Title: "居家练背，一根弹力带就够了") 平台：小红书；话题：健身；博主：在家练的老张；发布日期：2025-02-11；曝光量：596000；点赞数：48900；收藏数：55400；评论数：3600；简介：动作角度与组次安排 Doc 3(Title: "四十平出租屋改造，总共花了六千八") 平台：小红书；话题：家居；博主：租房也要好好住；发布日期：2025-03-28；曝光量：1260000；点赞数：108000；收藏数：132000；评论数：9700；简介：收纳、灯光、软装的花费明细</information><think> I still lack part of the information. </think><search> unknown 内容列表 </search><information>Doc 1(Title: "上岸之后我才明白的三件事") 平台：抖音；话题：考研；作者：考研过来人；发布日期：2025-05-13；播放量：3480000；点赞数：213000；分享数：71000；评论数：36000；时长：57秒；简介：信息差、心态与选择 Doc 2(Title: "这个隐藏关，百分之九十九的人没打过") 平台：抖音；话题：游戏；作者：游戏考古队；发布日期：2025-02-05；播放量：6670000；点赞数：389000；分享数：102000；评论数：52000；时长：88秒；简介：触发条件与路线 Doc 3(Title: "成分党看配方表，第一个要看的就是它") 平台：抖音；话题：美妆；作者：配方拆解阿宁；发布日期：2025-04-17；播放量：2910000；点赞数：176000；分享数：33000；评论数：19000；时长：73秒；简介：活性成分排序与浓度判断</information><think> Now I can answer based on the retrieved evidence. </think><answer> 厨房实验室 </answer>
```

</details>

**Q**: 哔哩哔哩上标题为《三个月徒手训练，我的体脂率从30%到8%》的内容，UP主是谁？

- gold: `['信息不足']`
- pred: `铁馆阿泽`
- 检索次数 2｜命中文档 ['bili_001', 'bili_002', 'bili_003', 'bili_007', 'bili_013']｜支撑文档 []
- 归因理由: 检索命中，但聚合/数值/多跳推理出错
- 触发行为: `ideal`

<details><summary>完整轨迹</summary>

```
<think> I need external information for this. </think><search> 哔哩哔哩上标题为《三个月徒手训练，我的体脂率从30%到8%》的内容，UP主是谁？ </search><information>Doc 1(Title: "三个月徒手训练，我的体脂率从26%到17%") 平台：哔哩哔哩；话题：健身；UP主：铁馆阿泽；发布日期：2025-01-25；播放量：2680000；点赞数：121000；投币数：78000；收藏数：34000；评论数：34000；时长：932秒；简介：训练计划与饮食控制的完整记录 Doc 2(Title: "三跨上岸某985，我的全年时间线") 平台：哔哩哔哩；话题：考研；UP主：上岸研究所；发布日期：2025-04-08；播放量：1350000；点赞数：67000；投币数：125000；收藏数：23000；评论数：23000；时长：1290秒；简介：公共课与专业课的时间分配表 Doc 3(Title: "2025年旗舰机横评：五台真机三个月的续航真相") 平台：哔哩哔哩；话题：数码测评；UP主：极客湾Geekerwan；发布日期：2025-03-12；播放量：3820000；点赞数：156000；投币数：91000；收藏数：42000；评论数：42000；时长：1180秒；简介：重点对比了续航与温控，结论是能效比优先于峰值性能</information><think> I still lack part of the information. </think><search> unknown 内容列表 </search><information>Doc 1(Title: "上岸之后我才明白的三件事") 平台：抖音；话题：考研；作者：考研过来人；发布日期：2025-05-13；播放量：3480000；点赞数：213000；分享数：71000；评论数：36000；时长：57秒；简介：信息差、心态与选择 Doc 2(Title: "这个隐藏关，百分之九十九的人没打过") 平台：抖音；话题：游戏；作者：游戏考古队；发布日期：2025-02-05；播放量：6670000；点赞数：389000；分享数：102000；评论数：52000；时长：88秒；简介：触发条件与路线 Doc 3(Title: "成分党看配方表，第一个要看的就是它") 平台：抖音；话题：美妆；作者：配方拆解阿宁；发布日期：2025-04-17；播放量：2910000；点赞数：176000；分享数：33000；评论数：19000；时长：73秒；简介：活性成分排序与浓度判断</information><think> Now I can answer based on the retrieved evidence. </think><answer> 铁馆阿泽 </answer>
```

</details>

## `gold_strict` — 语义正确但严格 EM 判错，属评测口径问题

共 1 例，下面取 3 例。

**Q**: 小红书「旅行」话题下一共有多少条内容？

- gold: `['2']`
- pred: `12`
- 检索次数 1｜命中文档 ['dy_004', 'xhs_010', 'xhs_011']｜支撑文档 ['xhs_010', 'xhs_011']
- 归因理由: 宽松口径（包含匹配）正确，严格 EM 判错
- 触发行为: `unfaithful`

<details><summary>完整轨迹</summary>

```
<think> Let me look this up. </think><search> 小红书「旅行」话题下一共有多少条内容？ </search><information>Doc 1(Title: "大理四天三晚，避开人从众的路线") 平台：小红书；话题：旅行；博主：慢旅行阿元；发布日期：2025-04-02；曝光量：783000；点赞数：66400；收藏数：52100；评论数：5900；简介：环海西路与古城外围的替代路线 Doc 2(Title: "一个人在京都住了七天，最推荐的三个街区") 平台：小红书；话题：旅行；博主：和风生活家；发布日期：2025-03-19；曝光量：441000；点赞数：37200；收藏数：31800；评论数：2800；简介：住宿选址与通勤时间取舍 Doc 3(Title: "一天走完一座小城，路线我排好了") 平台：抖音；话题：旅行；作者：周末去哪儿；发布日期：2025-03-22；播放量：6210000；点赞数：376000；分享数：69000；评论数：34000；时长：55秒；简介：时间分配与交通衔接</information><think> I remember the answer. </think><answer> 12 </answer>
```

</details>
