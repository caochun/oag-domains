from __future__ import annotations

import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
IMAGE_RE = re.compile(r"!\[[^\]]*]\(([^)]+)\)")
EVENT_RE = re.compile(
    r"(?P<idx>\d+)\.\s*"
    r"(?P<start>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s*至\s*"
    r"(?P<end>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)，"
    r"站点：(?P<station>[^，]+)，设备：(?P<device>[^，]+)，"
    r"事件类型：(?P<event_type>[^，]+)，二级摘要：(?P<summary>.*?)"
    r"，事件性质：(?P<nature>.*?)，告警等级：(?P<level>\d+)",
    re.S,
)
TIME_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?")
EVENT_START_PREFIX_RE = re.compile(r"(?:^|[\s。；;,，、])(?:\d{1,4}[.、)]\s*)?$")
FIELD_LABELS = [
    "站点",
    "设备",
    "事件类型",
    "二级摘要",
    "摘要",
    "事件性质",
    "告警等级",
]
FIELD_RE = re.compile(
    rf"({'|'.join(FIELD_LABELS)})\s*[：:]\s*(.*?)"
    rf"(?=(?:[，,；;]\s*)?(?:{'|'.join(FIELD_LABELS)})\s*[：:]|$)",
    re.S,
)

SYSTEM_ALIASES = {
    "dc_line": ["直流线路", "线路低电压", "线路故障", "故障穿越", "再启动"],
    "dc_control": ["直流站控", "功率控制", "电流控制", "模式顺序", "闭锁顺序"],
    "pole_control": ["极控", "极保护", "阀组控制", "VCE", "阀控制"],
    "ac_filter": ["交流滤波器", "滤波器", "低压电抗器", "无功控制"],
    "valve_cooling": ["阀冷", "内水冷", "冷却水", "主循环泵", "加药泵", "开关阀"],
    "station_power": ["站用电", "站用变", "降压变", "分接", "400V"],
    "recorder": ["录波", "故障录波", "录波启动"],
    "monitoring": ["系统监视", "测控", "自监视"],
}

EVIDENCE_QUERY_TERMS = {
    "dc_line": "直流线路 低电压保护 故障穿越 再启动 保护动作 复归",
    "dc_control": "功率控制 电流控制 自动功率 目标功率 变化速率 解锁 闭锁",
    "pole_control": "极控 阀组控制 VCE 控制保护系统 切换 A套 B套",
    "ac_filter": "交流滤波器 投退 无功控制 低压电抗器 开关 合闸 油泵",
    "valve_cooling": "阀冷 内水冷 主循环泵 开关阀 流量 温度 压力 异常 处理",
    "station_power": "站用电 站用变 分接开关 测控 自监视 轻微故障",
    "recorder": "故障录波 录波启动 事件记录 SER 保护动作",
    "monitoring": "系统监视 测控 自监视 告警 轻微故障",
}

NORMAL_TERMS = [
    "复归",
    "消失",
    "完成",
    "解锁动作完成",
    "合闸",
    "投入",
    "切换中信号消失",
    "油泵启动信号消失",
    "升降操作执行完毕",
]

ABNORMAL_TERMS = [
    "异常",
    "轻微故障",
    "故障告警",
    "自监视异常",
    "未储能",
    "指令下发存在异常",
    "响应不一致",
    "差异",
]

SEVERE_TERMS = [
    "保护动作",
    "低电压保护",
    "跳闸",
    "功率回降",
    "停运",
    "故障穿越",
    "再启动",
]

NEGATIVE_OUTCOME_TERMS = [
    "失败",
    "不成功",
    "未复归",
    "持续",
    "停运",
    "跳闸",
    "闭锁",
]

PRIMARY_KNOWLEDGE_ROLE = "emergency_procedure"
PRIMARY_KNOWLEDGE_BOOST = 100.0
HIGH_VALUE_EVIDENCE_BOOST = 8.0
STATION_EVIDENCE_BOOST = 35.0
EVENT_CONCLUSION_PREFIX = "conc_"
TOC_HEADINGS = {"目录", "目 录", "contents", "content"}
TOC_PENALTY = 80.0
PRIMARY_EVIDENCE_LIMIT = 4
KB_SCHEMA_VERSION = "3"
