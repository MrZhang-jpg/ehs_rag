#!/bin/bash
# 远端 FastAPI 服务管理：svc.sh start|stop|restart|status
# 说明：容器内无 ss/lsof，用 pgrep 定位进程（模式加 [ ] 防止匹配到本脚本自身）。
PORT=8000
PROJ=/root/autodl-tmp/ehs_rag_project/02_retrieval
PY=/root/.virtualenvs/ehs_rag/bin/python
LOG=/root/autodl-tmp/api_server.log
PAT='[0]4_api_server.py'

pid_of() { pgrep -f "$PAT" | head -1; }

is_up() { curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

do_stop() {
  P=$(pid_of)
  if [ -z "$P" ]; then
    is_up && { echo "WARN: 健康检查通但未找到进程，强制按端口清理失败"; return 1; }
    echo "NOT_RUNNING"; return 0
  fi
  kill "$P" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$P" 2>/dev/null || break; sleep 0.5; done
  kill -9 "$P" 2>/dev/null
  echo "STOPPED pid=$P"
}

do_start() {
  if is_up; then echo "ALREADY_UP pid=$(pid_of)"; return 0; fi
  cd "$PROJ" || exit 1
  nohup "$PY" 04_api_server.py > "$LOG" 2>&1 &
  for _ in $(seq 1 40); do
    is_up && { echo "STARTED pid=$(pid_of)"; return 0; }
    sleep 1
  done
  echo "START_FAILED"; tail -20 "$LOG"; return 1
}

case "$1" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; sleep 1; do_start ;;
  status)  if is_up; then echo "UP pid=$(pid_of)"; else echo "DOWN"; fi ;;
  *) echo "usage: svc.sh start|stop|restart|status" ;;
esac
