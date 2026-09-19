from dataclasses import dataclass

import pytest

from app.services.bond_engine import (
    HoldSpan,
    SeatCell,
    conflicts_with,
    contiguous_runs,
    find_bond_across_rows,
    find_contiguous_block,
)


def _row(cols, aisles=()):
    return [SeatCell(row=1, col=c, is_aisle=(c in aisles)) for c in cols]


def test_aisle_breaks_runs():
    cells = _row(range(1, 11), aisles={5, 6})
    assert contiguous_runs(cells) == [(1, 4), (7, 10)]


def test_find_contiguous_skips_occupied():
    cells = _row(range(1, 9))
    holds = [HoldSpan(row=1, start_col=2, end_col=3)]
    block = find_contiguous_block(cells, holds, 1, 3)
    assert block == HoldSpan(row=1, start_col=4, end_col=6)


def test_party_too_large_returns_none():
    cells = _row(range(1, 5), aisles={3})
    assert find_contiguous_block(cells, [], 1, 3) is None


def test_conflict_overlap():
    existing = [HoldSpan(row=2, start_col=4, end_col=6)]
    cand = HoldSpan(row=2, start_col=6, end_col=8)
    assert conflicts_with(existing, cand) == existing


def test_find_across_rows():
    seats = {
        1: _row(range(1, 5)),
        2: [SeatCell(row=2, col=c) for c in range(1, 9)],
    }
    holds = [HoldSpan(row=1, start_col=1, end_col=4)]
    block = find_bond_across_rows(seats, holds, 4)
    assert block == HoldSpan(row=2, start_col=1, end_col=4)


# ---------------------------------------------------------------------------
# 表驱动测例（金样 = 当前引擎语义；与注释不符的边界在 note 标「现网行为」）
# rows_spec: [(row, [col...], {aisle cols})]；holds: [(row, start, end)]
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunCase:
    name: str
    map_summary: str  # 厅图摘要
    cols: tuple
    aisles: frozenset
    expected: tuple  # [(start, end), ...]
    note: str = ""


RUN_CASES = [
    RunCase(
        "中段双过道切成两段",
        "1-10 座，5、6 列为过道",
        tuple(range(1, 11)), frozenset({5, 6}),
        ((1, 4), (7, 10)),
    ),
    RunCase(
        "单过道在尾部",
        "1-5 座，5 列为过道",
        tuple(range(1, 6)), frozenset({5}),
        ((1, 4),),
    ),
    RunCase(
        "单过道在开头",
        "1-5 座，1 列为过道",
        tuple(range(1, 6)), frozenset({1}),
        ((2, 5),),
    ),
    RunCase(
        "连续三段过道",
        "1-9 座，3、4、7 列为过道，余 1-2 / 5-6 / 8-9 三段",
        tuple(range(1, 10)), frozenset({3, 4, 7}),
        ((1, 2), (5, 6), (8, 9)),
    ),
    RunCase(
        "全过道无段",
        "1-3 全为过道",
        tuple(range(1, 4)), frozenset({1, 2, 3}),
        (),
    ),
    RunCase(
        "无过道整段",
        "1-6 连续座、无过道",
        tuple(range(1, 7)), frozenset(),
        ((1, 6),),
    ),
    RunCase(
        "乱序输入仍按列号排序",
        "传入列顺序 3,1,2，无过道",
        (3, 1, 2), frozenset(),
        ((1, 3),),
    ),
    RunCase(
        "列号缺口同样断段",
        "只有 1,2,4,5 四个座（3 列缺失而非过道），断成 1-2 与 4-5",
        (1, 2, 4, 5), frozenset(),
        ((1, 2), (4, 5)),
        note="现网行为：docstring 只称过道断段，实际列号不连续也会断段",
    ),
]


@pytest.mark.parametrize("case", RUN_CASES, ids=[c.name for c in RUN_CASES])
def test_contiguous_runs_table(case):
    cells = [SeatCell(row=1, col=c, is_aisle=(c in case.aisles)) for c in case.cols]
    assert contiguous_runs(cells) == list(case.expected), f"用例失败: {case.name}"


@dataclass(frozen=True)
class BlockCase:
    name: str
    map_summary: str  # 厅图摘要
    rows_spec: tuple  # ((row, cols, aisles), ...)
    holds: tuple  # ((row, start, end), ...) 已占区间
    party_size: int  # 人数
    expected: object  # HoldSpan 或 None
    note: str = ""


def _h(row, start, end):
    return HoldSpan(row=row, start_col=start, end_col=end)


BLOCK_CASES = [
    # --- 持座挖洞后连续段重建 ---
    BlockCase(
        "洞中挖断取左侧段",
        "单排 1-7；3-5 被占，左余 1-2、右余 6-7",
        ((1, tuple(range(1, 8)), ()),),
        ((1, 3, 5),), 2,
        _h(1, 1, 2),
        note="挖洞后重建连续段，最左够长的 1-2 命中",
    ),
    BlockCase(
        "左段不够跳过洞口取右段",
        "单排 1-6；2 被占，左余 1（1 座）、右余 3-6（4 座）",
        ((1, tuple(range(1, 7)), ()),),
        ((1, 2, 2),), 2,
        _h(1, 3, 4),
        note="挖洞后重建：左段 1 座不够 2 人，右段 3-6 命中 3-4",
    ),
    BlockCase(
        "两洞三段只取所需长度",
        "单排 1-9；2、6 被占，余 1 / 3-5 / 7-9 三段",
        ((1, tuple(range(1, 10)), ()),),
        ((1, 2, 2), (1, 6, 6)), 3,
        _h(1, 3, 5),
    ),
    # --- party_size 为 0 或超过整段 ---
    BlockCase(
        "人数为0直接无命中",
        "单排 1-5 全空",
        ((1, tuple(range(1, 6)), ()),),
        (), 0,
        None,
        note="party_size<=0 短路返回 None，即便整排全空",
    ),
    BlockCase(
        "人数超过整段长度无命中",
        "单排 1-4 全空，找 5 人",
        ((1, tuple(range(1, 5)), ()),),
        (), 5,
        None,
    ),
    BlockCase(
        "人数超过过道切出的最长段无命中",
        "1-6 座，4 列为过道，段为 1-3、5-6；找 4 人",
        ((1, tuple(range(1, 7)), (4,)),),
        (), 4,
        None,
    ),
    # --- 左右空段并存时的现网稳定结果（最左优先） ---
    BlockCase(
        "左右皆有空段取最左",
        "单排 1-8；4-5 被占，左余 1-3、右余 6-8；找 2 人",
        ((1, tuple(range(1, 9)), ()),),
        ((1, 4, 5),), 2,
        _h(1, 1, 2),
        note="现网策略：左右并存时稳定取最左连续段，且只取 party_size 长度",
    ),
    BlockCase(
        "左段不足退回右段",
        "单排 1-8；2-4 被占，左余 1（1 座）、右余 5-8（4 座）；找 3 人",
        ((1, tuple(range(1, 9)), ()),),
        ((1, 2, 4),), 3,
        _h(1, 5, 7),
        note="现网策略：最左段不够人数时稳定落到右侧段起点",
    ),
    # --- 跨排查找命中第一道够人数的排 ---
    BlockCase(
        "首排全满落到第二排",
        "排1: 1-4 全占；排2: 1-8 全空；找 4 人",
        (
            (1, tuple(range(1, 5)), ()),
            (2, tuple(range(1, 9)), ()),
        ),
        ((1, 1, 4),), 4,
        _h(2, 1, 4),
    ),
    BlockCase(
        "首排段不够长命中第二道够人的排",
        "排1: 1-4 带 3 列过道（最长段 2 座）；排2: 1-5 全空；找 3 人",
        (
            (1, tuple(range(1, 5)), (3,)),
            (2, tuple(range(1, 6)), ()),
        ),
        (), 3,
        _h(2, 1, 3),
        note="跨排按排号升序，第一道能容下 3 人的排为排2",
    ),
    BlockCase(
        "中间排先满足则不看后排",
        "排1: 1-2（仅 2 座）；排2: 1-6 全空；排3: 1-6 全空；找 4 人",
        (
            (1, tuple(range(1, 3)), ()),
            (2, tuple(range(1, 7)), ()),
            (3, tuple(range(1, 7)), ()),
        ),
        (), 4,
        _h(2, 1, 4),
    ),
    BlockCase(
        "字典乱序仍命中最小排号",
        "字典以 3,1 顺序给出；排1: 1-5 全空，排3: 1-5 全空；找 2 人",
        (
            (3, tuple(range(1, 6)), ()),
            (1, tuple(range(1, 6)), ()),
        ),
        (), 2,
        _h(1, 1, 2),
        note="现网行为：引擎按 sorted(row) 升序查找，与字典插入顺序无关",
    ),
    BlockCase(
        "各排均不够无命中",
        "排1: 1-2 带 2 列过道（仅座 1）；排2: 1-3 带 2 列过道（段 1、3）；找 2 人",
        (
            (1, (1, 2), (2,)),
            (2, (1, 2, 3), (2,)),
        ),
        (), 2,
        None,
    ),
]


@pytest.mark.parametrize("case", BLOCK_CASES, ids=[c.name for c in BLOCK_CASES])
def test_find_block_table(case):
    seats = {
        row: [SeatCell(row=row, col=c, is_aisle=(c in aisles)) for c in cols]
        for row, cols, aisles in case.rows_spec
    }
    holds = [_h(r, s, e) for r, s, e in case.holds]

    if len(seats) == 1:
        row = next(iter(seats))
        got = find_contiguous_block(seats[row], holds, row, case.party_size)
    else:
        got = find_bond_across_rows(seats, holds, case.party_size)

    assert got == case.expected, (
        f"用例失败: {case.name}｜厅图: {case.map_summary}｜"
        f"已占: {list(case.holds)}｜人数: {case.party_size}｜实得: {got}"
    )


@dataclass(frozen=True)
class ConflictCase:
    name: str
    map_summary: str  # 厅图摘要
    existing: tuple  # 已占区间 ((row, start, end), ...)
    candidate: tuple  # 候选区间 (row, start, end)
    expected_rows: tuple  # 命中的已占区间在 existing 中的下标
    note: str = ""


CONFLICT_CASES = [
    ConflictCase(
        "部分重叠命中",
        "已占 排2:4-6；候选 排2:5-7，在 5-6 部分重叠",
        ((2, 4, 6),), (2, 5, 7),
        (0,),
    ),
    ConflictCase(
        "右端点相接不重叠",
        "已占 排2:4-6；候选 排2:7-9，端点 6 与 7 相接",
        ((2, 4, 6),), (2, 7, 9),
        (),
        note="现网行为：端点相接（end+1 == start）视为不重叠、不冲突",
    ),
    ConflictCase(
        "左端点相接不重叠",
        "已占 排2:4-6；候选 排2:1-3，端点 3 与 4 相接",
        ((2, 4, 6),), (2, 1, 3),
        (),
        note="现网行为：端点相接视为不重叠、不冲突",
    ),
    ConflictCase(
        "共享单座即冲突",
        "已占 排2:4-6；候选 排2:6-8，共享 6 号座",
        ((2, 4, 6),), (2, 6, 8),
        (0,),
        note="现网行为：闭区间共享一个座位即判重叠",
    ),
    ConflictCase(
        "同区间完全重叠",
        "已占与候选均为 排2:4-6",
        ((2, 4, 6),), (2, 4, 6),
        (0,),
    ),
    ConflictCase(
        "候选包住已占",
        "已占 排2:4-6；候选 排2:1-10 包住",
        ((2, 4, 6),), (2, 1, 10),
        (0,),
    ),
    ConflictCase(
        "候选落在已占内部",
        "已占 排2:2-8；候选 排2:4-6 落在其内",
        ((2, 2, 8),), (2, 4, 6),
        (0,),
    ),
    ConflictCase(
        "不同排不冲突",
        "已占 排1:4-6；候选 排2:4-6，列相同但排不同",
        ((1, 4, 6),), (2, 4, 6),
        (),
    ),
    ConflictCase(
        "多条已占只返回重叠者",
        "已占 排2:1-2、4-6、8-9；候选 排2:5-8，与后两条各有重叠",
        ((2, 1, 2), (2, 4, 6), (2, 8, 9)), (2, 5, 8),
        (1, 2),
        note="现网行为：候选 5-8 同时与 4-6（共享 5,6）和 8-9（共享 8）重叠",
    ),
    ConflictCase(
        "夹在两条已占之间不冲突",
        "已占 排2:1-3、7-9；候选 排2:4-6，两侧均为端点相接",
        ((2, 1, 3), (2, 7, 9)), (2, 4, 6),
        (),
        note="现网行为：两侧端点相接均不视为重叠",
    ),
]


@pytest.mark.parametrize("case", CONFLICT_CASES, ids=[c.name for c in CONFLICT_CASES])
def test_conflicts_with_table(case):
    existing = [_h(r, s, e) for r, s, e in case.existing]
    candidate = _h(*case.candidate)
    got = conflicts_with(existing, candidate)
    expected = [existing[i] for i in case.expected_rows]
    assert got == expected, (
        f"用例失败: {case.name}｜厅图: {case.map_summary}｜"
        f"已占: {list(case.existing)}｜候选: {case.candidate}｜实得: {got}"
    )
