# AGENTS.md — Dobby 项目上下文（给 AI 助手看的记忆）

> 这份文件是从 Claude Code 的本地记忆导出的，供任何 AI 编码助手（Codex / Claude Code / 其他）在任何电脑上读取。
> 修改项目约定时请同步更新本文件。最后整理：2026-07-03。

## 用户是谁

- **Sagara**，GitHub `Sagara911`，邮箱 `huobingli0924@gmail.com`（站点展示用）。
- 中文为主，英文技术词混用（"playable"、"deploy"、"build command" 直接夹在中文句子里），回复请匹配这个语域。
- 主业是 H5 playable 广告开发，Windows + PowerShell 环境，熟 git CLI。
- 会问"该不该做 X"这类产品问题——要给带 trade-off 的真实建议，不要只回"可以，做法是……"。
- Cloudflare / GitHub 后台的 UI 操作用户自己点，助手负责推代码。

## 项目是什么

**Dobby**（原名 Playable Toolkit，2026-05-21 改名）——纯浏览器本地运行的素材处理工具站，文件不上传。
Logo 🧦，tagline「你的本地素材精灵」/「Dobby is free!」。

- **工具数：49**（2026-07-03 验证）。永远不要凭记忆报数——用 `grep -cE "href: 'tools/" assets/shared.js` 重新数。首页统计由 `Toolkit.TOOLS.length` 运行时计算，**不要重新引入硬编码数字**。
- 5 个分类：🖼️ image / 🎬 anim / 🔊 av / 🗜️ code / 📊 audit。工具清单的唯一真源是 `assets/shared.js` 的 `TOOLS` 数组。

### 关键架构

- `assets/shared.js` — Topbar、handoff（工具间结果传递）、主题、news toast、反馈 popover、i18n 引擎、所有共享 helper。
- `assets/i18n-strings.js` — zh/en 双语字典（2400+ key 对）。
- `assets/codecs.js` — 懒加载 WASM 编码器（mozjpeg / oxipng / webp-hq / AVIF，来自 esm.sh/@jsquash）。
- `assets/anim-encoders.js` — GIF/APNG 编解码。
- `assets/rife.js` + `assets/farneback.js` — 光流插帧（RIFE ONNX WebGPU，CPU 回退 Farnebäck）。
- `assets/esrgan.js` — AI 超分（TF.js WebGPU，模型自托管在 `models/`）。注意：ort-web WebGPU 跑不了 Real-ESRGAN（Clip op 崩溃），只有 TF.js 行。
- `sw.js` — Service Worker，**network-first**，缓存 key `CACHE = 'dobby-vN'`。
- 所有压缩输出必须做 magic-byte + 大小校验后才接受，防止静默损坏。

### 新增工具的固定流程

1. 在 `assets/shared.js` 注册：`TOOLS` 数组 + `INSTRUCTIONS` map +（如接收其他工具输出）`HANDOFF_ACCEPTS`。
2. 加进 `sw.js` precache 列表。
3. **任何用户可见文件改动都要 bump `sw.js` 的 `CACHE = 'dobby-vN'`**（当前 v129），否则老用户拿不到更新。
4. i18n：新增文案用 `data-i18n` 属性 / `Toolkit.T(key, vars, 中文fallback)`，zh + en 都写进 `assets/i18n-strings.js`。动态字符串**必须传中文 fallback**。

## 部署（push 即上线）

`git push origin main` 同时触发两路部署（各约 30 秒）：

1. **Cloudflare Pages（主站，国内快）**：https://dobby-aih.pages.dev/ — 项目名 `dobby`，`-aih` 后缀是 Cloudflare 防抢注机制，去不掉（除非买域名，用户已明确不买）。
2. **GitHub Pages（备份）**：https://sagara911.github.io/Dobby/ — 走 `.github/workflows/pages.yml`。

**死链接，永远不要引用**：`playable-toolkit.pages.dev`、`sagara911.github.io/playable-toolkit/`。

用户报"刷新没变化"→ 几乎都是 SW 缓存，先确认 `dobby-vN` 有没有 bump，再让用户 Ctrl+Shift+R。
终极手段（KILLSWITCH）：所有 HTML `<head>` 里有 `localStorage['toolkit-cleaned-vN']` 一次性清缓存脚本（当前 v2），只在用户设备真的卡死在旧 SW 上时才整体 bump。

Cloudflare Web Analytics token 在 `shared.js` `setupCloudflareAnalytics()`，localhost 等本地域名已排除。

## news.json 推送策略（重要，用户明确规定过）

- **用户可感知的改动**（新工具 / 新引擎 / bug 修复 / 新选项 / 重大 UX 改版）→ **同一次 push 里带上 news.json 条目**，不要单独问"要不要加"。
- **纯装饰 / 静态文案 / 内部重构 / CI 配置** → 不加条目。
- 判断标准：*"用户现在打开工具，不用别人告诉，自己会注意到变化吗？"* 会 → 加。
- 即使是当天刚上的功能再迭代，只要改动实质（布局重构、默认行为翻转）也算重大改动，要加。
- 一次 push 一个条目即可（多个功能打包 bullets），版本号 `dobby-vN` 是 news 自己的序列，**和 sw.js 的 CACHE 序列是两套独立编号，落后是正常的，不要"修"这个差距**。
- sw.js CACHE bump 是独立决策：只要用户可见文件变了就 bump，跟加不加 news 无关。

## Dobby 的文案人格（voice）

- 称呼用户为「**你**」或省略主语，**绝不用「主人」「您」「老板」**——Dobby is free，自由精灵没有主人。
- Dobby 第三人称自称：「Dobby 干完显示在这里」「丢文件进来，Dobby 给列清楚」。
- 固定短语保留：错误前缀 `Bad Dobby!`、tagline `Dobby is free!`、beta 标签 `Dobby 还在学本事`、logo 🧦。
- 英文版同样保持角色腔：「Dobby has come to help!」「Dobby will...」。

## 工作方式（用户给过的反馈，务必遵守）

1. **数字必须核实再说**。用户抓过"27 个工具"（实际 26）、"GitHub Pages 改名自动跳转"（实际 404）这类错。任何数量 / 版本号 / 外部系统行为，先查真源再报。
2. **诚实，不要 oversell**。有 trade-off（体积 vs 速度、质量 vs 兼容）就摆出来给用户选，不要替用户决定，也不要说"全是最优"这种大话。原话：「你得把好的接上，让用户自己去选择」。模式：默认 = 原生/轻量/快，可选引擎 = WASM/重/最好。
3. **视觉改版先出 mockup 再上线**。2026-05-27 试过 Eagle 风大卡片被用户否掉（「板块设计的不好看，还不如以前」）。49 个工具的仪表盘要密度和可扫描性，不要营销落地页式大色块。功能性工具可以直接做，纯视觉 CSS 改动先给看。

## 明确决定不做的（不要重新提议）

- AV1 视频编码（WASM 太重 + WebCodecs 支持不全）。
- libimagequant 调色板（现有 median-cut 够用）。
- 像素游戏工具包（只讨论过，用户说「只讨论，不动手」；若重开，建议做独立站共享 shared.js 基建，不并入 Dobby）。
- 自定义域名（用户评估后不买：「不买了，就这样」）。
- `dobby.pages.dev` 无后缀域名（Cloudflare 政策上不可能）。

## 已排队 / 未了事项

- **video-watermark 半透明 logo「alpha 反推真实还原」**——已同意做，主动搁置（风险最高，等单独 session）。
- 双语 news.json（目前条目纯中文，en 用户看英文外壳 + 中文内容）——除非用户提，否则跳过。
- 没有其他排队项。用户要新工具时问方向，可提的候选：glTF 优化器 / GLSL sandbox / CSV↔JSON / 文本 diff / CSS 生成器包。

## 相关但独立的本地项目：R1_01

`D:\Game\R1_01\`（不在本仓库）：改完 `index.html`（开发版）后**必须跑 `node build.js`** 生成 `R1.html`（投放版，内联 assets），看到「===== 打包完成 =====」才算完。这条只针对 R1_01，Dobby 走的是 push 自动部署。
